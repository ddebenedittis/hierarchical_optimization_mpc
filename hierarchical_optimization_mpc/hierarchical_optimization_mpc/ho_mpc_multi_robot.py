import copy
from dataclasses import dataclass, field
from enum import auto, Enum
import itertools

import casadi as ca
import numpy as np
from hierarchical_qp.hierarchical_qp import HierarchicalQP
from hierarchical_optimization_mpc.ho_mpc import subs, HOMPC
from scipy.special import binom



class TaskType(Enum):
    Same = auto()
    Sum = auto()
    SameTimeDiff = auto()
    Bi = auto()

# ============================================================================ #
#                                     HOMPC                                    #
# ============================================================================ #

class HOMPCMultiRobot(HOMPC):
    """
    Class to perform Model Predictive Control (MPC) with Hierarchical
    Optimization on a heterogeneous multi-robot system.
    """
    
    
    @dataclass
    class Task:
        """A single hierarchical task."""
        
        name:  str                  # task name
        prio:  int                  # task priority
        
        type: TaskType
        
        eq_task_ls: list[ca.SX]           # equality part: eq_task_ls = eq_coeff
        eq_coeff: list[list[np.ndarray]]
        ineq_task_ls: list[ca.SX]         # inequality part: ineq_task_ls = ineq_coeff
        ineq_coeff: list[list[np.ndarray]]
        
        eq_J_T_s: list[ca.SX] = field(repr=False)     # jacobian(eq_task_ls, state)
        eq_J_T_u: list[ca.SX] = field(repr=False)     # jacobian(eq_task_ls, input)
        ineq_J_T_s: list[ca.SX] = field(repr=False)   # jacobian(ineq_task_ls, state)
        ineq_J_T_u: list[ca.SX] = field(repr=False)   # jacobian(ineq_task_ls, input)
        
        aux_var: ca.SX = None
        mapping: list[ca.SX] = None
    
    # ======================================================================== #
    
    def __init__(
        self, states: list[ca.SX], inputs: list[ca.SX], fs: list[ca.SX], n_robots: list[int]
    ) -> None:
        """
        Initialize the instance.

        Args:
            states (list[ca.SX]): list of state symbolic variable for each robot class
            inputs (list[ca.SX]): list of input symbolic variable for each robot class
            fs (list[ca.SX]):     list of discrete-time system equations for each robot class:
                                  state_{k+1} = f(state_k, input_k)
            n_robots(list[int]):  list of the number of robots for each robot class
        """
        
        if len(states) != len(inputs) \
            or len(states) != len(fs) \
            or len(states) != len(n_robots):
            raise ValueError(
                "states, inputs, fs, and n_robots do not have the same size. " +
                "Their size must be equal to the number of robot classes."
            )
        
        for i, n_r in enumerate(n_robots):
            if n_r < 0:
                raise ValueError(f'The {i}-th class of robots has a negative number of robots.')
        
        self._n_control = 1 # control horizon timesteps
        self._n_pred = 0    # prediction horizon timesteps (the input is constant)
        
        self.regularization = 1e-6  # regularization factor
        
        # ==================================================================== #
        
        self._states = states   # state variable
        self._inputs = inputs   # input variable
        
        # Number of robots for every robot class.
        self.n_robots = n_robots
        
        # State and input variables of every robot class.
        self._n_states: list[int] = [state.numel() for state in states]
        self._n_inputs: list[int] = [input.numel() for input in inputs]
        
        # System models: state_{k+1} = f(state_k, input_k)
        self._models = [
            ca.Function(
                'f',
                [states[i], inputs[i]],
                [fs[i]],
                ['state', 'input'],
                ['state_kp1']
            ) for i in range(len(states))
        ]
        
        self._Js_f_x: list[ca.SX] = [ca.jacobian(fs[i], states[i]) for i in range(len(fs))]
        self._Js_f_u: list[ca.SX] = [ca.jacobian(fs[i], inputs[i]) for i in range(len(fs))]
        
        # States around which the linearization is performed.
        # _state_bar[class c][robot j][timestep k]
        self._state_bar = [
            [[None] * (self.n_control + self.n_pred)] * n_robots[i]
            for i in range(len(states))
        ]
        # Inputs around which the linearization is performed.
        # _input_bar[class c][robot j][timestep k]
        self._input_bar = [
            [[np.zeros(self._n_inputs[i])] * self.n_control] * n_robots[i]
            for i in range(len(states))
        ]
        
        self._tasks: list[self.Task] = []
                
    # =========================== Class Properties =========================== #
    
    @property
    def n_control(self):
        return self._n_control
    
    @n_control.setter
    def n_control(self, value):
        if value < 1:
            ValueError('"n_control" must be equal or greater than 1.')
        else:
            self._n_control = value
            
            # Adapt the sizes of the state and input linearization points.
            self._state_bar = [
                [[None] * (self.n_control + self.n_pred)] * self.n_robots[i]
                for i in range(len(self.n_robots))
            ]
            self._input_bar = [
                [[np.zeros(self._n_inputs[i]) for _ in range(self.n_control)] for _ in range(self.n_robots[i])]
                for i in range(len(self.n_robots))
            ]
    @property
    def n_pred(self):
        return self._n_pred
    
    @n_pred.setter
    def n_pred(self, value):
        if value < 0:
            ValueError('"n_pred" must be equal or greater than 0.')
        else:
            self._n_pred = value
            
            # Adapt the size of the state linearization points.
            self._state_bar = [
                [[None] * (self.n_control + self.n_pred)] * self.n_robots[i]
                for i in range(len(self.n_robots))
            ]
        
    # ============================== Initialize ============================== #
        
    def _initialize(
        self,
        states_meas: list[list[np.ndarray]] = None,
        inputs: list[list[list[np.ndarray]]] = None):
        """
        Update the linearization points (x_bar_k, u_bar_k) from the current
        position and the history of optimal inputs.
        When state_meas in None, rerun the optimization updating the
        linearization points. When state_meas is given, either perform the
        optimization linearing around the evolution with the previous optimal
        inputs shifted by one (when inputs is None), or with the given inputs
        sequence.

        Args:
            state_meas (np.ndarray): measured state.
            inputs (list[np.ndarray]): optimal inputs previously computed.
        """
        
        n_c = self._n_control
        n_p = self._n_pred
        
        if states_meas is None:
            # Rerun the optimization. The trajectory is linearized around the
            # previous optimal inputs.
            states_meas = [[self._state_bar[c][j][0] for j in range(self.n_robots[c])] for c in range(len(self.n_robots))]
        elif inputs is None:
            # Shift the previous optimal inputs by one.
            for c, n_r in enumerate(self.n_robots):
                for j in range(n_r):
                    self._input_bar[c][j][0:-1] = self._input_bar[c][j][1:]
        else:
            # Use the given inputs.
            self._input_bar = inputs
                
        # Generate the state linearization points from the measured state and
        # the linearization inputs.
        for i, n_r in enumerate(self.n_robots):
            model = self._models[i]
            for j in range(n_r):
                states_meas[i][j] = states_meas[i][j].reshape((-1, 1))    # convert to a column vector.
                self._state_bar[i][j] = [states_meas[i][j]] * (n_c + n_p + 1)
                for k in range(self._n_control):       
                    self._state_bar[i][j][k+1] = model(
                        self._state_bar[i][j][k], self._input_bar[i][j][k]
                    ).full()
                
                # When in the prediction phase, the system input is constant and equal
                # to the last input.
                for k in range(self._n_pred):
                    self._state_bar[i][j][k+n_c] = model(
                        self._state_bar[i][j][n_c-1], self._input_bar[i][j][n_c-1]
                    ).full()
        
    # ============================ Linearize_model =========================== #
    
    def _linearize_model(
        self, robot_class: int, state_bar: np.ndarray, input_bar: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the matrices A_k, B_k, and f_bar_k of the system linearized in
        (state_bar, input_bar), i.e.
        state_k+1 = A_k state_tilde_k + B_k u_tilde_k + f_bar_k + x_k_bar

        Args:
            state_bar (np.ndarray): _description_
            input_bar (np.ndarray): _description_

        Returns:
            [np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        
        A = subs(
            [self._Js_f_x[robot_class]],
            [self._states[robot_class], self._inputs[robot_class]],
            [state_bar, input_bar]
        )
        A = A
        
        B = subs(
            [self._Js_f_u[robot_class]],
            [self._states[robot_class], self._inputs[robot_class]],
            [state_bar, input_bar]
        )
        
        f_bar = self._models[robot_class](state_bar, input_bar).full()
        
        return A, B, f_bar
    
    # ======================= Dynamics_consistency_task ====================== #
    
    def _task_dynamics_consistency(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct the constraints matrices that enforce the system dynamics.
        The task equation is
            s_tilde_k+1 = A_k s_tilde_k + B_k u_tilde_k
            
        Where s_tilde_k = s_k - s_bar_k, s_k is the state, and s_bar_k is the
        state linearization point.

        (x_tilde_0 is a zero vector)
        """
        
        n_classes = len(self.n_robots)
        
        n_s = self._n_states
        
        n_c = self._n_control
        n_p = self._n_pred
        
        n_x_opt = self._get_n_x_opt()
        
        n_rows = sum(np.multiply(self.n_robots, n_s) * (n_c+n_p))
        
        A_dyn = np.zeros((n_rows, n_x_opt))
        b_dyn = np.zeros((n_rows, 1))
        
        i = 0   # row index used for the matrices A_dyn and b_dyn.
        for c in range(n_classes):
            n_s = self._n_states[c]
            for j in range(self.n_robots[c]):
                # For the first timestep it is different because state_tilde_k is 0.
                A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, 0)] = np.eye(n_s)
                [_, B_0, _] = self._linearize_model(c, self._state_bar[c][j][0], self._input_bar[c][j][0])
                A_dyn[i:i+n_s, self._get_idx_input_k(c, j, 0)] = - B_0
                i += n_s
                    
                # Formulate the task for the remaining timesteps.
                for k in range(1, n_c + n_p):
                    A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, k)] = np.eye(n_s)
                    [A_k, B_k, _] = self._linearize_model(c, self._state_bar[c][j][k], self._input_bar[c][j][k])
                    A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, k-1)] = - A_k
                    A_dyn[i:i+n_s, self._get_idx_input_k(c, j, k)] = - B_k
                    
                    i += n_s
            
        return A_dyn, b_dyn
            
    # ============================== Create_task ============================= #
            
    def create_task(
        self,
        name: str,
        prio: int,
        type: TaskType,
        eq_task_ls: list[ca.SX] = None,
        eq_task_coeff: list[list[np.ndarray]] = None,
        ineq_task_ls: list[ca.SX] = None,
        ineq_task_coeff: list[list[np.ndarray]] = None,
    ):
        """
        Create a HOMPC.Task

        Args:
            name (str): task name
            prio (int): task priority
            eq_task_ls (ca.SX, optional): _description_.
            eq_task_coeff (list[np.ndarray], optional): _description_.
            ineq_task_ls (ca.SX, optional): _description_.
            ineq_task_coeff (list[np.ndarray], optional): _description_.
        """
        
        if eq_task_ls is None:
            eq_task_ls = [ca.SX.sym("eq", 0)] * len(self.n_robots)
            
        if ineq_task_ls is None:
            ineq_task_ls = [ca.SX.sym("ineq", 0)] * len(self.n_robots)
                
        if eq_task_coeff is None:
            eq_task_coeff = [self.InfList([np.zeros((eq_task_ls[c].size1(),1))]) for c in range(len(self.n_robots))]
            
        if ineq_task_coeff is None:
            ineq_task_coeff = [self.InfList([np.zeros((ineq_task_ls[c].size1(),1))]) for c in range(len(self.n_robots))]
        
        self._tasks.append(self.Task(
            name = name,
            prio = prio,
            type = type,
            eq_task_ls = eq_task_ls,
            eq_J_T_s = [ca.jacobian(eq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))],
            eq_J_T_u = [ca.jacobian(eq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))],
            eq_coeff = [self.InfList([e.reshape((-1, 1)) for e in eq_task_coeff[c]]) for c in range(len(self.n_robots))],
            ineq_task_ls = ineq_task_ls,
            ineq_J_T_s = [ca.jacobian(ineq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))],
            ineq_J_T_u = [ca.jacobian(ineq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))],
            ineq_coeff = [self.InfList([e.reshape((-1, 1)) for e in ineq_task_coeff[c]]) for c in range(len(self.n_robots))]
        ))
        
    # ======================================================================== #
    
    def create_task_bi(
        self,
        name: str,
        prio: int,
        type: TaskType,
        aux: ca.SX,
        mapping: list[ca.SX] = None,
        eq_task_ls: ca.SX = None,
        eq_task_coeff: list[np.ndarray] = None,
        ineq_task_ls: ca.SX = None,
        ineq_task_coeff: list[np.ndarray] = None,
    ):
        if eq_task_ls is None:
            eq_task_ls = ca.SX.sym("eq", 0)
        if ineq_task_ls is None:
            ineq_task_ls = ca.SX.sym("ineq", 0)
        
        self._tasks.append(self.Task(
            name = name,
            prio = prio,
            type = type,
            eq_task_ls = eq_task_ls,
            eq_J_T_s = None,
            eq_J_T_u = None,
            eq_coeff = None,
            ineq_task_ls = ineq_task_ls,
            ineq_J_T_s = None,
            ineq_J_T_u = None,
            ineq_coeff = None,
            aux_var = aux,
            mapping = mapping,
        ))
    
    # ======================================================================== #
        
    def _create_task_i_matrices(self, ie) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the matrices that are used for the HO MPC problem.

        Args:
            ie (_type_): _description_

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: [A, b, C, d]
        """
        
        t = self._tasks[ie]
        n_c = self._n_control
        n_p = self._n_pred
        
        if t.type == TaskType.Same:
            ne = sum(np.multiply(
                [t.eq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                self.n_robots,
            )) * (n_c + n_p)
            
            ni = sum(np.multiply(
                [t.ineq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                self.n_robots,
            )) * (n_c + n_p)
            
            A = np.zeros((ne, self._get_n_x_opt()))
            b = np.zeros((ne, 1))
            C = np.zeros((ni, self._get_n_x_opt()))
            d = np.zeros((ni, 1))
            
            ie = 0
            ii = 0
            for c, n_r in enumerate(self.n_robots):
                for j in range(n_r):
                    for k in range(n_c + n_p):
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]
                        
                        [
                            A[ie:ie+ne, self._get_idx_state_kp1(c, j, k)],
                            A[ie:ie+ne, self._get_idx_input_k(c, j, k)],
                            b[ie:ie+ne],
                            C[ii:ii+ni, self._get_idx_state_kp1(c, j, k)],
                            C[ii:ii+ni, self._get_idx_input_k(c, j, k)],
                            d[ii:ii+ni],
                        ] = self._helper_create_task_i_matrices(t, c, j, k)
                        
                        ie += ne
                        ii += ni
                        
        # ==================================================================== #
        
        elif t.type == TaskType.Sum:
            ne = t.eq_J_T_s[0].shape[0] * (n_c + n_p)
            
            ni = t.ineq_J_T_s[0].shape[0] * (n_c + n_p)
            
            A = np.zeros((ne, self._get_n_x_opt()))
            b = np.zeros((ne, 1))
            C = np.zeros((ni, self._get_n_x_opt()))
            d = np.zeros((ni, 1))
            
            for c, n_r in enumerate(self.n_robots):
                for j in range(n_r):
                    for k in range(n_c + n_p):                            
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]
                        
                        [
                            A[k*ne:(k+1)*ne, self._get_idx_state_kp1(c, j, k)],
                            A[k*ne:(k+1)*ne, self._get_idx_input_k(c, j, k)],
                            b_temp,
                            C[k*ni:(k+1)*ni, self._get_idx_state_kp1(c, j, k)],
                            C[k*ni:(k+1)*ni, self._get_idx_input_k(c, j, k)],
                            d_temp,
                        ] = self._helper_create_task_i_matrices(t, c, j, k)
                        
                        b[k*ne:(k+1)*ne] += b_temp
                        d[k*ni:(k+1)*ni] += d_temp
                        
        # ==================================================================== #
        
        elif t.type == TaskType.SameTimeDiff:
            ne = sum(np.multiply(
                [t.eq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                self.n_robots,
            )) * (n_c + n_p)
            
            ni = sum(np.multiply(
                [t.ineq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                self.n_robots,
            )) * (n_c + n_p)
            
            A = np.zeros((ne, self._get_n_x_opt()))
            b = np.zeros((ne, 1))
            C = np.zeros((ni, self._get_n_x_opt()))
            d = np.zeros((ni, 1))
            
            ie = 0
            ii = 0
            for c, n_r in enumerate(self.n_robots):
                for j in range(n_r):
                    for k in range(1, n_c + n_p):
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]
                        
                        [
                            A[ie:ie+ne, self._get_idx_state_kp1(c, j, k)],
                            A[ie:ie+ne, self._get_idx_input_k(c, j, k)],
                            b[ie:ie+ne],
                            C[ii:ii+ni, self._get_idx_state_kp1(c, j, k)],
                            C[ii:ii+ni, self._get_idx_input_k(c, j, k)],
                            d[ii:ii+ni],
                        ] = self._helper_create_task_i_matrices(t, c, j, k)
                        
                        [
                            A[ie:ie+ne, self._get_idx_state_kp1(c, j, k-1)],
                            A[ie:ie+ne, self._get_idx_input_k(c, j, k-1)],
                            b[ie:ie+ne],
                            C[ii:ii+ni, self._get_idx_state_kp1(c, j, k-1)],
                            C[ii:ii+ni, self._get_idx_input_k(c, j, k-1)],
                            d[ii:ii+ni],
                        ] = [-e for e in self._helper_create_task_i_matrices(t, c, j, k-1)]
                        
                        if k > self.n_control - 1:
                            ki = self.n_control - 1
                        else:
                            ki = k
                        
                        b[ie:ie+ne] = t.eq_coeff[c][k] - subs(
                            [t.eq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        ) + subs(
                            [t.eq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k], self._input_bar[c][j][ki-1]],
                        )
                        
                        d[ii:ii+ni] = t.ineq_coeff[c][k] - subs(
                            [t.ineq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        ) + subs(
                            [t.ineq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k], self._input_bar[c][j][ki-1]],
                        )
                        
                        ie += ne
                        ii += ni
                        
        elif t.type == TaskType.Bi:
            ne = (int)(binom(sum(self.n_robots), 2) * t.eq_task_ls.size1()) * (n_c + n_p)
            
            ni = (int)(binom(sum(self.n_robots), 2) * t.ineq_task_ls.size1()) * (n_c + n_p)
                        
            A = np.zeros((ne, self._get_n_x_opt()))
            b = np.zeros((ne, 1))
            C = np.zeros((ni, self._get_n_x_opt()))
            d = np.zeros((ni, 1))
            
            ie = 0
            ii = 0
            for e_c in set(itertools.combinations([i for i in range(len(self.n_robots))] * 2, 2)):
                if e_c[0] == e_c[1]:
                    c = e_c[0]
                    for e_j in set(itertools.combinations(
                        [i for i in range(self.n_robots[c])], 2
                    )):
                        j0 = e_j[0]
                        j1 = e_j[1]
                                                
                        ne = (int)(binom(sum(self.n_robots), 2) * t.eq_task_ls.size1())
                        
                        ni = (int)(binom(sum(self.n_robots), 2) * t.ineq_task_ls.size1())
                        
                        for k in range(2, n_c + n_p):
                            [
                                A[ie:ie+ne, self._get_idx_state_kp1(c, j0, k)],
                                A[ie:ie+ne, self._get_idx_state_kp1(c, j1, k)],
                                A[ie:ie+ne, self._get_idx_input_k(c, j0, k)],
                                A[ie:ie+ne, self._get_idx_input_k(c, j1, k)],
                                b[ie:ie+ne],
                                C[ii:ii+ni, self._get_idx_state_kp1(c, j0, k)],
                                C[ii:ii+ni, self._get_idx_state_kp1(c, j1, k)],
                                C[ii:ii+ni, self._get_idx_input_k(c, j0, k)],
                                C[ii:ii+ni, self._get_idx_input_k(c, j1, k)],
                                d[ii:ii+ni],
                            ] = self._helper_create_task_bi_i_matrices(t, c, j0, c, j1, k)
                                                        
                            ie += ne
                            ii += ni
                else:
                    c0 = e_c[0]
                    c1 = e_c[1]
                    for e_j in list(itertools.product(
                        [i for i in range(self.n_robots[c0])],
                        [i for i in range(self.n_robots[c1])]
                    )):
                        j0 = e_j[0]
                        j1 = e_j[1]
                        
                        ne = t.eq_task_ls.shape[0]
                        ni = t.ineq_task_ls.shape[0]
                        
                        for k in range(2, n_c + n_p):
                            [
                                A[ie:ie+ne, self._get_idx_state_kp1(c0, j0, k)],
                                A[ie:ie+ne, self._get_idx_state_kp1(c1, j1, k)],
                                A[ie:ie+ne, self._get_idx_input_k(c0, j0, k)],
                                A[ie:ie+ne, self._get_idx_input_k(c1, j1, k)],
                                b[ie:ie+ne],
                                C[ii:ii+ni, self._get_idx_state_kp1(c0, j0, k)],
                                C[ii:ii+ni, self._get_idx_state_kp1(c1, j1, k)],
                                C[ii:ii+ni, self._get_idx_input_k(c0, j0, k)],
                                C[ii:ii+ni, self._get_idx_input_k(c1, j1, k)],
                                d[ii:ii+ni],
                            ] = self._helper_create_task_bi_i_matrices(t, c0, j0, c1, j1, k)
                            
                            ie += ne
                            ii += ni
            
        return A, b, C, d
    
    # ==================== _helper_create_task_i_matrices ==================== #
    
    # Auxiliary matrix to create the matrices A, b, C, d
    def _helper_create_task_i_matrices(
        self, t: Task, c: int, j: int, k: int
    ):
        """
        _summary_

        Args:
            t (Task): task
            c (int): robot class index
            j (int): robot number in the class
            k (int): timestep

        Returns:
            [
                A[e1:e2, self._get_idx_state_kp1(c, j, k)],
                A[e1:e2, self._get_idx_input_k(c, j, k)],
                b[e1:e2],
                C[i1:i2, self._get_idx_state_kp1(c, j, k)],
                C[i1:i2, self._get_idx_input_k(c, j, k)],
                d[i1:i2],
            ]
        """
        
        if k > self.n_control - 1:
            ki = self.n_control - 1
        else:
            ki = k
        
        return [
            subs(
                [t.eq_J_T_s[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
            ),
            subs(
                [t.eq_J_T_u[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
            ),
            t.eq_coeff[c][k] - subs(
                [t.eq_task_ls[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
            ),
            subs(
                [t.ineq_J_T_s[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
            ),
            subs(
                [t.ineq_J_T_u[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
            ),
            t.ineq_coeff[c][k] - subs(
                [t.ineq_task_ls[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
            ),
        ]
        
    # ======================================================================== #
    
    # Auxiliary matrix to create the matrices A, b, C, d
    def _helper_create_task_bi_i_matrices(
        self, t: Task,
        c0: int, j0: int,
        c1: int, j1: int,
        k: int,
    ):
        """
        _summary_

        Args:
            t (Task): task
            c (int): robot class index
            j (int): robot number in the class
            k (int): timestep

        Returns:
            [
                A[e1:e2, self._get_idx_state_kp1(c1, j1, k)],
                A[e1:e2, self._get_idx_state_kp1(c2, j2, k)],
                A[e1:e2, self._get_idx_input_k(c1, j1, k)],
                A[e1:e2, self._get_idx_input_k(c2, j2, k)],
                b[e1:e2],
                C[i1:i2, self._get_idx_state_kp1(c1, j1, k)],
                C[i1:i2, self._get_idx_state_kp1(c2, j2, k)],
                C[i1:i2, self._get_idx_input_k(c1, j1, k)],
                C[i1:i2, self._get_idx_input_k(c2, j2, k)],
                d[i1:i2],
            ]
        """
        
        if k > self.n_control - 1:
            ki = self.n_control - 1
        else:
            ki = k
                
        def aux(self, task_ls: ca.SX, ci: int, derivating_var: ca.SX, c0: int = c0, c1: int = c1):
            if ci == c0:
                i = 0
            else:
                i = 1
            
            return subs(
                [ca.jacobian(task_ls, t.aux_var[i,:]) @ ca.jacobian(t.mapping[ci], derivating_var[ci])],
                [self._states[ci], self._inputs[ci], ca.vertcat(t.aux_var[0,:].T), ca.vertcat(t.aux_var[1,:].T)],
                [self._state_bar[ci][j0][k+1], self._input_bar[ci][j0][ki],
                 subs([t.mapping[c0]], [self._states[c0], self._inputs[c0]], [self._state_bar[c0][j0][k+1], self._input_bar[c0][j0][ki]]),
                 subs([t.mapping[c1]], [self._states[c1], self._inputs[c1]], [self._state_bar[c1][j1][k+1], self._input_bar[c1][j1][ki]])]
            ),
        
        return [
            aux(t.eq_task_ls, c0, self._states),
            aux(t.eq_task_ls, c1, self._states),
            aux(t.eq_task_ls, c0, self._inputs),
            aux(t.eq_task_ls, c1, self._inputs),
            - subs(
                [t.eq_task_ls],
                [ca.vertcat(t.aux_var[0,:].T), ca.vertcat(t.aux_var[1,:].T)],
                [subs([t.mapping[c0]], [self._states[c0], self._inputs[c0]], [self._state_bar[c0][j0][k+1], self._input_bar[c0][j0][ki]]),
                 subs([t.mapping[c1]], [self._states[c1], self._inputs[c1]], [self._state_bar[c1][j1][k+1], self._input_bar[c1][j1][ki]])]
            ),
            
            aux(t.ineq_task_ls, c0, self._states),
            aux(t.ineq_task_ls, c1, self._states),
            aux(t.ineq_task_ls, c0, self._inputs),
            aux(t.ineq_task_ls, c1, self._inputs),
            - subs(
                [t.ineq_task_ls],
                [ca.vertcat(t.aux_var[0,:].T), ca.vertcat(t.aux_var[1,:].T)],
                [subs([t.mapping[c0]], [self._states[c0], self._inputs[c0]], [self._state_bar[c0][j0][k+1], self._input_bar[c0][j0][ki]]),
                 subs([t.mapping[c1]], [self._states[c1], self._inputs[c1]], [self._state_bar[c1][j1][k+1], self._input_bar[c1][j1][ki]])]
            ),
        ]
        
    # ======================================================================== #
    
    def __call__(self, state_meas: np.ndarray = None, inputs: list[np.ndarray] = None) -> np.ndarray:
        self._initialize(state_meas, inputs)
        
        n_tasks = len(self._tasks)
        
        A = [None] * (1 + n_tasks)
        b = [None] * (1 + n_tasks)
        C = [None] * (1 + n_tasks)
        d = [None] * (1 + n_tasks)
        
        A[0], b[0] = self._task_dynamics_consistency()
        
        for k in range(n_tasks):
            kp = k + 1
            A[kp], b[kp], C[kp], d[kp] = self._create_task_i_matrices(k)
                                    
        hqp = HierarchicalQP()
        
        x_star = hqp(A, b, C, d)
        
        n_c = self._n_control
        
        u_0 = [
            [self._input_bar[c][j][0] + x_star[self._get_idx_input_k(c, j, 0)]
                for j in range(self.n_robots[c])]
            for c in range(len(self.n_robots))
        ]
        
        for c, n_r in enumerate(self.n_robots):
            for j in range(n_r):
                for k in range(n_c):
                    self._input_bar[c][j][k] = copy.deepcopy(self._input_bar[c][j][k] + x_star[self._get_idx_input_k(c, j, k)])
                
        return u_0
    
    # ======================================================================== #
    
    def _get_n_x_opt(self) -> int:
        """
        Return the dimension of the optimization vector n_x_opt.
        n_x_opt = [u_0; u_1; ...; u_{n_c-1}; s_1; s_2; ...; s_{n_c+n_p}]
        """
        
        n_x = 0
        for i, n_r in enumerate(self.n_robots):
            n_x += n_r * (
                self.n_control * self._n_inputs[i] \
                    + (self.n_control + self.n_pred) * self._n_states[i]
            )
        
        return n_x
    
    def _get_idx_input_k(self, c: int, j: int, k: int) -> np.ndarray:
        """
        Return the indices that correspond to the input u_{c, j, k} in n_x_opt.
        When k >= n_c, return u_{c, j, n_c-1}.

        Args:
            c (int): robot class index
            j (int): robot number in the class c
            k (int): timestep k

        Returns:
            np.ndarray: _description_
        """
        
        if k < 0 or k > self._n_control + self._n_pred - 1:
            raise ValueError
        
        if k > self._n_control - 1:
            k = self._n_control - 1
            
        n_c = self._n_control  
        n_i = self._n_inputs[c]
        
        temp1 = sum(np.multiply(self.n_robots[0:c], self._n_inputs[0:c])) * n_c
        temp2 = j * n_i * n_c
        
        return np.arange(
            temp1 + temp2 + k*n_i,
            temp1 + temp2 + (k+1)*n_i
        )
    
    def _get_idx_state_kp1(self, c: int, j: int, k: int) -> np.ndarray:
        """Return the indices that correspond to the state s_{k+1} in n_x_opt."""
        
        if j < 0 or j > self.n_robots[c] - 1:
            raise ValueError
        
        if k < 0 or k > self._n_control + self._n_pred - 1:
            raise ValueError
        
        n_c = self._n_control
        n_s = self._n_states[c]
        
        temp1 = sum(np.multiply(self.n_robots, self._n_inputs)) * n_c
        temp2 = sum(np.multiply(self.n_robots[0:c], self._n_states[0:c])) * n_c
        temp3 = j * n_s * n_c
        
        if j == -1:
            return np.arange(
            temp1 + temp2 + temp3 + k * n_s,
            temp1 + temp2 + temp3 + (k+1) * n_s
        )
        
        return np.arange(
            temp1 + temp2 + temp3 + k * n_s,
            temp1 + temp2 + temp3 + (k+1) * n_s
        )
        