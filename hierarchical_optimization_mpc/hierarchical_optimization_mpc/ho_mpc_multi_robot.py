from dataclasses import dataclass, field
from enum import auto, Enum

import casadi as ca
import numpy as np
from hierarchical_qp.hierarchical_qp import HierarchicalQP
from hierarchical_optimization_mpc.ho_mpc import subs, HOMPC



class TaskType(Enum):
    Same = auto()
    Sum = auto()

# ============================================================================ #
#                                     HOMPC                                    #
# ============================================================================ #

class HOMPCMultiRobot(HOMPC):
    """
    Class to perform Model Predictive Control (MPC) with Hierarchical
    Optimization.
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

    
    # ======================================================================== #
    
    def __init__(self, states: list[ca.SX], inputs: list[ca.SX], fs: list[ca.SX], n_robots: list[int]):
        """
        Initialize the instance.

        Args:
            state (ca.SX): state symbolic variable
            input (ca.SX): input symbolic variable
            f (ca.SX):     discrete-time system equation:
                           state_{k+1} = f(state_k, input_k)
        """
        
        
        
        self._n_control = 1 # control horizon timesteps
        self._n_pred = 0    # prediction horizon timesteps (the input is constant)
        
        self.regularization = 1e-6  # regularization factor.
        
        # ==================================================================== #
        
        self._states = states # state variable
        self._inputs = inputs # input variable
        
        self.n_robots = n_robots
        
        self._n_states: list[int] = [state.numel() for state in states]
        self._n_inputs: list[int] = [input.numel() for input in inputs]
        
        # System model: state_{k+1} = f(state_k, input_k)
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
        
        self._state_bar = [[[None] * (self.n_control + self.n_pred)] * n_robots[i] for i in range(len(states))]
        self._input_bar = [
            [[np.zeros(self._n_inputs[i])] * self.n_control] * n_robots[i] for i in range(len(states))
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
            self._state_bar = [[[None] * (self.n_control + self.n_pred)] * self.n_robots[i] for i in range(len(self.n_robots))]
            self._input_bar = [
                [[np.zeros(self._n_inputs[i])] * self.n_control] * self.n_robots[i] for i in range(len(self.n_robots))
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
            self._state_bar = [[[None] * (self.n_control + self.n_pred)] * self.n_robots[i] for i in range(len(self.n_robots))]
        
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
            # use the given inputs.
            self._input_bar = inputs
                
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
        A = A # + np.eye(*A.shape)
        
        B = subs(
            [self._Js_f_u[robot_class]],
            [self._states[robot_class], self._inputs[robot_class]],
            [state_bar, input_bar]
        )
        
        f_bar = self._models[robot_class](state_bar, input_bar).full()
        
        return A, B, f_bar
    
    # ======================= Dynamics_consistency_task ====================== #
    
    def _task_dynamics_consistency(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the constraints matrices that enforce the system dynamics"""
        
        n_classes = len(self.n_robots)
        
        n_s = self._n_states
        
        n_c = self._n_control
        n_p = self._n_pred
        
        n_x_opt = self._get_n_x_opt()
        
        n_rows = self._get_idx_state_kp1(n_classes-1, self.n_robots[n_classes-1] - 1, n_c + n_p - 1)[-1] \
            - self._get_idx_state_kp1(0, 0, 0)[0] + 1
        
        A_dyn = np.zeros((n_rows, n_x_opt))
        b_dyn = np.zeros((n_rows, 1))
        
        i = 0
        
        for c in range(n_classes):
            n_s = self._n_states[c]
            for j in range(self.n_robots[c]):
                A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, 0)] = np.eye(n_s)
                
                [_, B_0, _] = self._linearize_model(c, self._state_bar[c][j][0], self._input_bar[c][j][0])
                A_dyn[i:i+n_s, self._get_idx_input_k(c, j, 0)] = - B_0
                i+= n_s
                    
                for k in range(1, n_c + n_p):
                    A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, k)] = np.eye(n_s)
                    
                    [A_k, B_k, _] = self._linearize_model(c, self._state_bar[c][j][k], self._input_bar[c][j][k])
                    A_dyn[i:i+n_s, self._get_idx_state_kp1(c, j, k-1)] = - A_k
                    A_dyn[i:i+n_s, self._get_idx_input_k(c, j, k)] = - B_k
                    
                    i+= n_s
            
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
        
    def create_task_i_matrices(self, ie) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                        if k > n_c - 1:
                            ki = n_c - 1
                        else:
                            ki = k
                            
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]
                        
                        A[ie:ie+ne, self._get_idx_state_kp1(c, j, k)] = subs(
                            [t.eq_J_T_s[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
                        )
                        A[ie:ie+ne, self._get_idx_input_k(c, j, k)] = subs(
                            [t.eq_J_T_u[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        b[ie:ie+ne] = t.eq_coeff[c][k] - subs(
                            [t.eq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        
                        C[ii:ii+ni, self._get_idx_state_kp1(c, j, k)] = subs(
                            [t.ineq_J_T_s[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
                        )
                        C[ii:ii+ni, self._get_idx_input_k(c, j, k)] = subs(
                            [t.ineq_J_T_u[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        d[ii:ii+ni] = t.ineq_coeff[c][k] - subs(
                            [t.ineq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        
                        ie += ne
                        ii += ni
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
                        if k > n_c - 1:
                            ki = n_c - 1
                        else:
                            ki = k
                            
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]
                        
                        A[k*ne:(k+1)*ne, self._get_idx_state_kp1(c, j, k)] = subs(
                            [t.eq_J_T_s[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
                        )
                        A[k*ne:(k+1)*ne, self._get_idx_input_k(c, j, k)] = subs(
                            [t.eq_J_T_u[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        b[k*ne:(k+1)*ne] = t.eq_coeff[c][k] - subs(
                            [t.eq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        
                        C[k*ni:(k+1)*ni, self._get_idx_state_kp1(c, j, k)] = subs(
                            [t.ineq_J_T_s[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]]
                        )
                        C[k*ni:(k+1)*ni, self._get_idx_input_k(c, j, k)] = subs(
                            [t.ineq_J_T_u[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
                        d[k*ni:(k+1)*ni] = t.ineq_coeff[c][k] - subs(
                            [t.ineq_task_ls[c]],
                            [self._states[c], self._inputs[c]],
                            [self._state_bar[c][j][k+1], self._input_bar[c][j][ki]],
                        )
            
        return A, b, C, d
        
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
            A[kp], b[kp], C[kp], d[kp] = self.create_task_i_matrices(k)
                                    
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
                    self._input_bar[c][j][k] = self._input_bar[c][j][k] + x_star[self._get_idx_input_k(c, j, k)]
                
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
        
        return np.arange(
            temp1 + temp2 + temp3 + k * n_s,
            temp1 + temp2 + temp3 + (k+1) * n_s
        )
        