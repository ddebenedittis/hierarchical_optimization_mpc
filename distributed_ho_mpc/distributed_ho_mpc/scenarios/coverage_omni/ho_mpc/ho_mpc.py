from dataclasses import dataclass, field

import casadi as ca
import numpy as np
from distributed_ho_mpc.scenarios.coverage_omni.ho_mpc.hierarchical_qp import HierarchicalQP, QPSolver
    


def subs(f: ca.SX, input: list[ca.SX], input_0: list[ca.SX]) -> np.ndarray:
    """Return f(input = input_0)."""
    
    fun = ca.Function(
        "f",
        input,
        f
    )
        
    return fun(*input_0).full()


# ============================================================================ #
#                                     HOMPC                                    #
# ============================================================================ #

class HOMPC:
    """
    Class to perform Model Predictive Control (MPC) with Hierarchical
    Optimization.
    
    The input at time i is u_i. The state at time i is s_i.
    
    The linearization points are (u_bar_i, s_bar_i).
    
    The optimization vector is
    [u_tilde_0, ..., u_tilde_n_c-1, s_tilde_1, ..., s_tilde_n_c]^T,\\
    where u_tilde_i = u_i - u_bar and s_tilde_i = s_i - s_bar.
    
    The system dynamics are x_tilde_k+1 = A_k x_tilde_k + B_k u_tilde_k.
    """
    
    
    class InfList(list):
        """
        "Infinite" mutable sequence. When accessing an index out of bounds, it
        returns the last item in the list.
        """
        
        def __getitem__(self, key):
            if key >= len(self):
                key = len(self) - 1
                    
            return super(HOMPC.InfList, self).__getitem__(key)
    
    
    @dataclass
    class Task:
        """A single hierarchical task."""
        
        name:  str                  # task name
        prio:  int                  # task priority
        
        eq_task_ls: ca.SX           # equality part: eq_task_ls = eq_coeff
        eq_coeff: list[np.ndarray]
        ineq_task_ls: ca.SX         # inequality part: ineq_task_ls = ineq_coeff
        ineq_coeff: list[np.ndarray]
        
        eq_J_T_s: ca.SX = field(repr=False)     # jacobian(eq_task_ls, state)
        eq_J_T_u: ca.SX = field(repr=False)     # jacobian(eq_task_ls, input)
        ineq_J_T_s: ca.SX = field(repr=False)   # jacobian(ineq_task_ls, state)
        ineq_J_T_u: ca.SX = field(repr=False)   # jacobian(ineq_task_ls, input)
    
    # ======================================================================== #
    
    def __init__(self, state: ca.SX, input: ca.SX, f: ca.SX, solver: QPSolver = QPSolver.quadprog):
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
        
        self.solver = solver
        
        # ==================================================================== #
        
        self._state = state # state variable
        self._input = input # input variable
        
        self._n_state: int = self._state.numel()
        self._n_input = self._input.numel()
        
        # System model: state_{k+1} = f(state_k, input_k)
        self._model = ca.Function(
            'f',
            [state, input],
            [f],
            ['state', 'input'],
            ['state_kp1']
        )
        
        self._J_f_x: ca.SX = ca.jacobian(f, state)
        self._J_f_u: ca.SX = ca.jacobian(f, input)
        
        self._state_bar = None
        self._input_bar = [np.zeros(self._n_input)] * self.n_control
        
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
            self._input_bar = [np.zeros(self._n_input)] * self.n_control
            
    @property
    def n_pred(self):
        return self._n_pred
    
    @n_pred.setter
    def n_pred(self, value):
        if value < 0:
            ValueError('"n_pred" must be equal or greater than 0.')
        else:
            self._n_pred = value
        
    # ============================== Initialize ============================== #
        
    def _initialize(self, state_meas: np.ndarray, inputs: list[np.ndarray] = None):
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
        
        if state_meas is None:
            # Rerun the optimization. The trajectory is linearized around the
            # previous optimal inputs.
            state_meas = self._state_bar[0]
        elif inputs is None:
            # Shift the previous optimal inputs by one.
            self._input_bar[0:-1] = self._input_bar[1:]
        else:
            # use the given inputs.
            self._input_bar = inputs
        
        state_meas = state_meas.reshape((-1, 1))    # convert to a column vector.
        self._state_bar = [state_meas] * (n_c + n_p + 1)
        for i in range(self._n_control):
            self._state_bar[i + 1] = self._model(
                self._state_bar[i], self._input_bar[i]
            ).full()
        
        # When in the prediction phase, the system input is constant and equal
        # to the last input.
        for i in range(self._n_pred):
            self._state_bar[n_c + i] = self._model(
                self._state_bar[n_c - 1], self._input_bar[n_c - 1]
            ).full()
                
    # ============================ Linearize_model =========================== #
    
    def _linearize_model(
        self, state_bar: np.ndarray, input_bar: np.ndarray
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
            [self._J_f_x],
            [self._state, self._input],
            [state_bar, input_bar]
        )
        A = A # + np.eye(*A.shape)
        
        B = subs(
            [self._J_f_u],
            [self._state, self._input],
            [state_bar, input_bar]
        )
        
        f_bar = self._model(state_bar, input_bar).full()
        
        return A, B, f_bar
    
    # ======================= Dynamics_consistency_task ====================== #
    
    def _task_dynamics_consistency(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the constraints matrices that enforce the system dynamics"""
        
        n_s = self._n_state
        
        n_c = self._n_control
        n_p = self._n_pred
        
        n_x_opt = self._get_n_x_opt()
        
        A_dyn = np.zeros(((n_c+n_p)*n_s, n_x_opt))
        b_dyn = np.zeros(((n_c+n_p)*n_s, 1))
        
        [_, B_0, _] = self._linearize_model(self._state_bar[0], self._input_bar[0])
        
        A_dyn[0:n_s, self._get_idx_state_kp1(0)] = np.eye(n_s)
        A_dyn[0:n_s, self._get_idx_input_k(0)] = - B_0
        
        for k in range(1, n_c+n_p):
            A_dyn[k*n_s:(k+1)*n_s, self._get_idx_state_kp1(k)] = np.eye(n_s)
            
            [A_k, B_k, _] = self._linearize_model(self._state_bar[k], self._input_bar[k-1])
            A_dyn[k*n_s:(k+1)*n_s, self._get_idx_state_kp1(k-1)] = - A_k
            A_dyn[k*n_s:(k+1)*n_s, self._get_idx_input_k(k)] = - B_k
            
        return A_dyn, b_dyn
            
    # ============================== Create_task ============================= #
            
    def create_task(
        self,
        name: str,
        prio: int,
        eq_task_ls: ca.SX = ca.SX.sym("eq", 0),
        eq_task_coeff: list[np.ndarray] = None,
        ineq_task_ls: ca.SX = ca.SX.sym("ineq", 0),
        ineq_task_coeff: list[np.ndarray] = None,
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
        
        if eq_task_coeff is None:
            eq_task_coeff = self.InfList([np.zeros((eq_task_ls.size1(),1))])
            
        if ineq_task_coeff is None:
            ineq_task_coeff = self.InfList([np.zeros((ineq_task_ls.size1(),1))])
        
        self._tasks.append(self.Task(
            name = name,
            prio = prio,
            eq_task_ls = eq_task_ls,
            eq_J_T_s = ca.jacobian(eq_task_ls, self._state),
            eq_J_T_u = ca.jacobian(eq_task_ls, self._input),
            eq_coeff = self.InfList([e.reshape((-1, 1)) for e in eq_task_coeff]),
            ineq_task_ls = ineq_task_ls,
            ineq_J_T_s = ca.jacobian(ineq_task_ls, self._state),
            ineq_J_T_u = ca.jacobian(ineq_task_ls, self._input),
            ineq_coeff = self.InfList([e.reshape((-1, 1)) for e in ineq_task_coeff])
        ))
        
    # ======================================================================== #
        
    def _create_task_i_matrices(self, i) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = self._tasks[i]
        ne = t.eq_J_T_s.shape[0]
        ni = t.ineq_J_T_s.shape[0]
        
        n_c = self._n_control
        n_p = self._n_pred
        
        A = np.zeros((ne * (n_c+n_p), self._get_n_x_opt()))
        b = np.zeros((ne * (n_c+n_p), 1))
        C = np.zeros((ni * (n_c+n_p), self._get_n_x_opt()))
        d = np.zeros((ni * (n_c+n_p), 1))
        
        for k in range(n_c + n_p):
            if k > n_c - 1:
                ki = n_c - 1
            else:
                ki = k
            
            A[k*ne:(k+1)*ne, self._get_idx_state_kp1(k)] = subs(
                [t.eq_J_T_s],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]]
            )
            A[k*ne:(k+1)*ne, self._get_idx_input_k(k)] = subs(
                [t.eq_J_T_u],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]],
            )
            b[k*ne:(k+1)*ne] = t.eq_coeff[k] - subs(
                [t.eq_task_ls],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]],
            )
            
            C[k*ni:(k+1)*ni, self._get_idx_state_kp1(k)] = subs(
                [t.ineq_J_T_s],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]]
            )
            C[k*ni:(k+1)*ni, self._get_idx_input_k(k)] = subs(
                [t.ineq_J_T_u],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]],
            )
            d[k*ni:(k+1)*ni] = t.ineq_coeff[k] - subs(
                [t.ineq_task_ls],
                [self._state, self._input],
                [self._state_bar[k+1], self._input_bar[ki]],
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
            A[kp], b[kp], C[kp], d[kp] = self._create_task_i_matrices(k)
                        
        hqp = HierarchicalQP(solver=self.solver)
        
        x_star = hqp(A, b, C, d)
        
        n_c = self._n_control
        
        u_0 = self._input_bar[0] + x_star[self._get_idx_input_k(0)]
        
        for k in range(n_c):
            self._input_bar[k] = self._input_bar[k] + x_star[self._get_idx_input_k(k)]
                
        return u_0
                        
    # ======================================================================== #
    
    def _get_n_x_opt(self) -> int:
        """
        Return the dimension of the optimization vector n_x_opt.
        n_x_opt = [u_0; u_1; ...; u_{n_c-1}; s_1; s_2; ...; s_{n_c+n_p}]
        """
        
        return self._n_control * self._n_input \
            + (self._n_control + self._n_pred) * self._n_state
    
    def _get_idx_input_k(self, k) -> np.ndarray:
        """
        Return the indices that correspond to the input u_{k} in n_x_opt.
        When k >= n_c, return u_{n_c-1}.
        """
        
        if k < 0 or k > self._n_control + self._n_pred - 1:
            raise ValueError
        
        if k > self._n_control - 1:
            k = self._n_control - 1
        
        n_i = self._n_input
        
        return np.arange(k*n_i, (k+1)*n_i)
    
    def _get_idx_state_kp1(self, k) -> np.ndarray:
        """Return the indices that correspond to the state s_{k+1} in n_x_opt."""
        
        if k < 0 or k > self._n_control + self._n_pred - 1:
            raise ValueError
        
        n_s = self._n_state
        n_i = self._n_input
        
        n_c = self._n_control
        
        return np.arange(
            n_c*n_i + k * n_s,
            n_c*n_i + (k+1) * n_s
        )
        