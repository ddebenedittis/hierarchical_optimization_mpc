import copy
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import casadi as ca
import numpy as np
from scipy.special import binom

from distributed_ho_mpc.ho_mpc.hierarchical_qp import (
    HierarchicalQP,
    QPSolver,
)
from distributed_ho_mpc.ho_mpc.ho_mpc import HOMPC, subs

np.set_printoptions(threshold=np.inf)

stack = True


class TaskIndexes(Enum):
    All = auto()
    Last = auto()


class TaskType(Enum):
    Same = auto()
    Sum = auto()
    SameTimeDiff = auto()
    Bi = auto()


class TaskBiCoeff:
    def __init__(self, c0, j0, c1, j1, k, coeff) -> None:
        self.c0 = c0
        self.j0 = j0
        self.c1 = c1
        self.j1 = j1
        self.k = k
        self.coeff = coeff

    def get(self):
        """
        Return the task coefficients:
            c0, j0, c1, j1, k, coeff
        """

        return self.c0, self.j0, self.c1, self.j1, self.k, self.coeff


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

        name: str  # task name
        prio: int  # task priority

        type: TaskType

        eq_task_ls: list[ca.SX]  # equality part: eq_task_ls = eq_coeff
        eq_coeff: list[list[list[np.ndarray]]]  # eq_coeff[c][j][k]
        ineq_task_ls: list[ca.SX]  # inequality part: ineq_task_ls = ineq_coeff
        ineq_coeff: list[list[list[np.ndarray]]]  # ineq_coeff[c][j][k]

        eq_J_T_s: list[ca.SX] = field(repr=False)  # jacobian(eq_task_ls, state)
        eq_J_T_u: list[ca.SX] = field(repr=False)  # jacobian(eq_task_ls, input)
        ineq_J_T_s: list[ca.SX] = field(repr=False)  # jacobian(ineq_task_ls, state)
        ineq_J_T_u: list[ca.SX] = field(repr=False)  # jacobian(ineq_task_ls, input)

        eq_weight: float = 1.0
        ineq_weight: float = 1.0

        aux_var: ca.SX = None
        mapping: list[ca.SX] = None

        time_index: TaskIndexes = TaskIndexes.All
        robot_index: list[list[int]] = None

    class ConstraintType(Enum):
        Both = auto()
        Eq = auto()
        Ineq = auto()

    # ======================================================================== #

    def __init__(
        self,
        states: list[ca.SX],
        inputs: list[ca.SX],
        fs: list[ca.SX],
        n_robots: list[int],
        n_neigh: int,
        solver: QPSolver = QPSolver.quadprog,
        hierarchical: bool = True,
        decay_rate: float = 1.0,
    ) -> None:
        """
        Initialize the instance.

        Args:
            states (list[ca.SX]): list of state symbolic variable for each robot class
            inputs (list[ca.SX]): list of input symbolic variable for each robot class
            fs (list[ca.SX]):     list of discrete-time system equations for each robot class:
                                  state_{k+1} = f(state_k, input_k)
            n_robots (list[int]): list of the number of robots for each robot class
            hierarchical (bool):  flag to enable hierarchical optimization
            decay_rate (float, optional): tasks decay rate for weighted approach (hierarchical == False). Between 0.0 and 1.0.
        """

        if len(states) != len(inputs) or len(states) != len(fs) or len(states) != len(n_robots):
            raise ValueError(
                'states, inputs, fs, and n_robots do not have the same size. '
                + 'Their size must be equal to the number of robot classes.'
            )

        for i, n_r in enumerate(n_robots):
            if n_r < 0:
                raise ValueError(f'The {i}-th class of robots has a negative number of robots.')

        self._n_control = 1  # control horizon timesteps
        self._n_pred = 0  # prediction horizon timesteps (the input is constant)

        self.regularization = 1e-6  # regularization factor

        self.solver = QPSolver.get_enum(solver)

        self.hierarchical = hierarchical

        self.decay_rate = decay_rate

        self.hqp = HierarchicalQP(solver=self.solver, hierarchical=self.hierarchical)
        self.hqp.ns = 3
        self.hqp.ni = 2

        # ==================================================================== #

        self._states = states  # state variable
        self._inputs = inputs  # input variable

        # Number of robots for every robot class.
        self.n_robots = n_robots
        self.degree = n_neigh  # number of neighbours

        # State and input variables of every robot class.
        self._n_states: list[int] = [state.numel() for state in states]
        self._n_inputs: list[int] = [input.numel() for input in inputs]

        # System models: state_{k+1} = f(state_k, input_k)
        self._models = [
            ca.Function('f', [states[i], inputs[i]], [fs[i]], ['state', 'input'], ['state_kp1'])
            for i in range(len(states))
        ]

        self._Js_f_x: list[ca.SX] = [ca.jacobian(fs[i], states[i]) for i in range(len(fs))]
        self._Js_f_u: list[ca.SX] = [ca.jacobian(fs[i], inputs[i]) for i in range(len(fs))]

        # States around which the linearization is performed.
        # _state_bar[class c][robot j][timestep k]
        self._state_bar = [
            [[None] * (self.n_control + self.n_pred)] * n_robots[i] for i in range(len(states))
        ]
        # Inputs around which the linearization is performed.
        # _input_bar[class c][robot j][timestep k]
        self._input_bar = [
            [[np.zeros(self._n_inputs[i])] * self.n_control] * n_robots[i]
            for i in range(len(states))
        ]

        self._tasks: list[self.Task] = []

        self.solve_times = {
            'Create Problem': 0,
            'Solve Problem': 0,
        }

    # =========================== Class Properties =========================== #

    @property
    def n_control(self):
        return self._n_control

    @n_control.setter
    def n_control(self, value):
        if value < 1:
            raise ValueError('"n_control" must be equal or greater than 1.')

        self._n_control = value

        # Adapt the sizes of the state and input linearization points.
        self._state_bar = [
            [[None] * (self.n_control + self.n_pred)] * self.n_robots[i]
            for i in range(len(self.n_robots))
        ]
        self._input_bar = [
            [
                [np.zeros(self._n_inputs[i]) for _ in range(self.n_control)]
                for _ in range(self.n_robots[i])
            ]
            for i in range(len(self.n_robots))
        ]

    @property
    def n_pred(self):
        return self._n_pred

    @n_pred.setter
    def n_pred(self, value):
        if value < 0:
            raise ValueError('"n_pred" must be equal or greater than 0.')

        self._n_pred = value

        # Adapt the size of the state linearization points.
        self._state_bar = [
            [[None] * (self.n_control + self.n_pred)] * self.n_robots[i]
            for i in range(len(self.n_robots))
        ]

    # ======================================================================== #

    def null_consensus_start(self):  # NOTE activate flag for Z intersection
        self.hqp.start_consensus = True

    def remove_robots(self, idx_robots: dict[list[int]]):
        """
        Remove robots from the optimization problem.

        Args:
            idx_robots (dict[list[int]]): dictionary of the indices of the robots to be removed.
        """

        for c, js in enumerate(idx_robots):
            self.n_robots[c] -= len(js)

            for j in js:
                self._state_bar[c].pop(j)
                self._input_bar[c].pop(j)

    # ============================== Initialize ============================== #

    def _initialize(
        self,
        states_meas: list[list[np.ndarray]] | None = None,
        inputs: list[list[list[np.ndarray]]] | None = None,
    ):
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
            states_meas = [
                [self._state_bar[c][j][0] for j in range(self.n_robots[c])]
                for c in range(len(self.n_robots))
            ]
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
                states_meas[i][j] = states_meas[i][j].reshape(
                    (-1, 1)
                )  # convert to a column vector.
                self._state_bar[i][j] = [states_meas[i][j]] * (n_c + n_p + 1)
                for k in range(self._n_control):
                    self._state_bar[i][j][k + 1] = model(
                        self._state_bar[i][j][k], self._input_bar[i][j][k]
                    ).full()

                # When in the prediction phase, the system input is constant and equal
                # to the last input.
                for k in range(self._n_pred):
                    self._state_bar[i][j][k + n_c] = model(
                        self._state_bar[i][j][n_c - 1], self._input_bar[i][j][n_c - 1]
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
            [state_bar, input_bar],
        )

        B = subs(
            [self._Js_f_u[robot_class]],
            [self._states[robot_class], self._inputs[robot_class]],
            [state_bar, input_bar],
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

        n_rows = sum(np.multiply(self.n_robots, n_s) * (n_c + n_p))

        # Example:
        # x_opt = [  s1   u0   s2   u1   s3   u2]^T
        #
        #         [   I -B_0    0    0    0    0]
        # A_dyn = [-A_1    0    I -B_1    0    0]
        #         [   0    0 -A_2    0    I -B_2]
        #
        # b_dyn = [0 0 0]^T

        A_dyn = np.zeros((n_rows, n_x_opt))
        b_dyn = np.zeros((n_rows, 1))

        i = 0  # row index used for the matrices A_dyn and b_dyn.
        for c in range(n_classes):
            n_s = self._n_states[c]
            for j in range(self.n_robots[c]):
                # For the first timestep it is different because state_tilde_k is 0.
                A_dyn[i : i + n_s, self._get_idx_state_kp1(c, j, 0)] = np.eye(n_s)
                [_, B_0, _] = self._linearize_model(
                    c, self._state_bar[c][j][0], self._input_bar[c][j][0]
                )
                A_dyn[i : i + n_s, self._get_idx_input_k(c, j, 0)] = -B_0
                i += n_s

                # Formulate the task for the remaining timesteps:
                for k in range(1, n_c + n_p):
                    A_dyn[i : i + n_s, self._get_idx_state_kp1(c, j, k)] = np.eye(n_s)
                    [A_k, B_k, _] = self._linearize_model(
                        c, self._state_bar[c][j][k], self._input_bar[c][j][k]
                    )
                    A_dyn[i : i + n_s, self._get_idx_state_kp1(c, j, k - 1)] = -A_k
                    A_dyn[i : i + n_s, self._get_idx_input_k(c, j, k)] = -B_k

                    i += n_s

        return A_dyn, b_dyn

    def add_robots(self, n_robots: list[int], states_meas: list[list[np.ndarray]]):
        """
        Modify the n_robots, _state_bar, and _input_bar attributes when adding robots.

        Args:
            n_robots (list[int]): number of robots to be added for each class
            states_meas (list[list][np.ndarray]): measured state of the added robots.

        Raises:
            ValueError: _description_
        """

        for c, n_r in enumerate(self.n_robots):
            if n_robots[c] < 0:
                raise ValueError(f'The {c}-th class of robots has a negative number of robots.')
            n_r += n_robots[c]
            _state_bar_new = [
                [[None] * (self.n_control + self.n_pred)] * n_robots[c]
                for i in range(len(self._states))
            ]
            _input_bar_new = [
                [np.zeros(self._n_inputs[i])] * self.n_control * n_robots[c]
                for i in range(len(self._inputs))
            ]

            self._state_bar[c].extend(_state_bar_new)
            self._input_bar[c].extend(_input_bar_new)

            self.n_robots[c] = n_r

    # ============================== Create_task ============================= #

    def create_task(
        self,
        name: str,
        prio: int,
        type: TaskType,
        eq_task_ls: list[ca.SX] | None = None,
        eq_task_coeff: list[list[list[np.ndarray]]] | None = None,
        eq_weight: float = 1.0,
        ineq_task_ls: list[ca.SX] | None = None,
        ineq_task_coeff: list[list[list[np.ndarray]]] | None = None,
        ineq_weight: float = 1.0,
        time_index: TaskIndexes = TaskIndexes.All,
        robot_index: list[list[int]] | None = None,
    ):
        """
        Create a HOMPC.Task.

        Args:
            name (str): task name
            prio (int): task priority
            eq_task_ls (ca.SX, optional): left side of the equality task.
            eq_task_coeff (list[np.ndarray], optional): coefficients on the right
                                                        side of the equality task.
            ineq_task_ls (ca.SX, optional): left side of the inequality task.
            ineq_task_coeff (list[np.ndarray], optional): coefficients on the right.
                                                          side of the in equality task.
        """

        if eq_task_ls is None:
            eq_task_ls = [ca.SX.sym('eq', 0)] * len(self.n_robots)

        if ineq_task_ls is None:
            ineq_task_ls = [ca.SX.sym('ineq', 0)] * len(self.n_robots)

        if robot_index is not None:
            if eq_task_coeff is not None:
                eq_coeff = [
                    [None for j in range(self.n_robots[c])] for c in range(len(self.n_robots))
                ]

                for c in range(len(robot_index)):
                    for idx, j in enumerate(robot_index[c]):
                        eq_coeff[c][j] = self.InfList(
                            [e.reshape((-1, 1)) for e in eq_task_coeff[c][idx]]
                        )

                eq_task_coeff = eq_coeff

            if ineq_task_coeff is not None:
                ineq_coeff = [
                    [None for j in range(self.n_robots[c])] for c in range(len(self.n_robots))
                ]

                for c in range(len(robot_index)):
                    for idx, j in enumerate(robot_index[c]):
                        ineq_coeff[c][j] = self.InfList(
                            [i.reshape((-1, 1)) for i in ineq_task_coeff[c][idx]]
                        )

                ineq_task_coeff = ineq_coeff

        self._tasks.append(
            self.Task(
                name=name,
                prio=prio,
                type=type,
                eq_task_ls=eq_task_ls,
                eq_J_T_s=[
                    ca.jacobian(eq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))
                ],
                eq_J_T_u=[
                    ca.jacobian(eq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))
                ],
                eq_coeff=None
                if eq_task_coeff is None
                else [
                    [
                        self.InfList(
                            [e.reshape((-1, 1)) for e in eq_task_coeff[c][j]]
                            if eq_task_coeff[c][j] is not None
                            else [None]
                        )
                        for j in range(self.n_robots[c])
                    ]
                    for c in range(len(self.n_robots))
                ],
                eq_weight=eq_weight,
                ineq_task_ls=ineq_task_ls,
                ineq_J_T_s=[
                    ca.jacobian(ineq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))
                ],
                ineq_J_T_u=[
                    ca.jacobian(ineq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))
                ],
                ineq_coeff=None
                if ineq_task_coeff is None
                else [
                    [
                        self.InfList(
                            [e.reshape((-1, 1)) for e in ineq_task_coeff[c][j]]
                            if ineq_task_coeff[c][j] is not None
                            else [None]
                        )
                        for j in range(self.n_robots[c])
                    ]
                    for c in range(len(self.n_robots))
                ],
                ineq_weight=ineq_weight,
                time_index=time_index,
                robot_index=robot_index,
            )
        )

    # ============================== Update_task ============================= #

    def update_task(
        self,
        name: str,
        prio: int | None = None,
        type: TaskType | None = None,
        eq_task_ls: list[ca.SX] | None = None,
        eq_task_coeff: list[list[list[np.ndarray]]] | None = None,
        ineq_task_ls: list[ca.SX] | None = None,
        ineq_task_coeff: list[list[list[np.ndarray]]] | None = None,
        time_index: TaskIndexes | None = None,
        robot_index: TaskIndexes | None = None,
        pos: int | None = None,
    ):
        for i, t in enumerate(self._tasks):
            if t.name == name:
                if pos is not None and i == pos:
                    id = i
                    break
                elif pos is None:
                    id = i
                    break

        if robot_index is not None:
            if eq_task_coeff is not None:
                eq_coeff = [
                    [[None] for j in range(self.n_robots[c])] for c in range(len(self.n_robots))
                ]

                for c in range(len(robot_index)):
                    for idx, j in enumerate(robot_index[c]):
                        eq_coeff[c][j] = self.InfList(
                            [e.reshape((-1, 1)) for e in eq_task_coeff[c][idx]]
                        )

                eq_task_coeff = eq_coeff

            if ineq_task_coeff is not None:
                ineq_coeff = [
                    [[None] for j in range(self.n_robots[c])] for c in range(len(self.n_robots))
                ]

                for c in range(len(robot_index)):
                    for idx, j in enumerate(robot_index[c]):
                        ineq_coeff[c][j] = self.InfList(
                            [i.reshape((-1, 1)) for i in ineq_task_coeff[c][idx]]
                        )

                ineq_task_coeff = ineq_coeff

        if prio is None:
            prio = self._tasks[id].prio
        if type is None:
            type = self._tasks[id].type
        if eq_task_ls is None:
            eq_task_ls = self._tasks[id].eq_task_ls
        if eq_task_coeff is None:
            eq_task_coeff = self._tasks[id].eq_coeff
        if ineq_task_ls is None:
            ineq_task_ls = self._tasks[id].ineq_task_ls
        if ineq_task_coeff is None:
            ineq_task_coeff = self._tasks[id].ineq_coeff
        if time_index is None:
            time_index = self._tasks[id].time_index
        if robot_index is None:
            robot_index = self._tasks[id].robot_index

        self._tasks[i] = self.Task(
            name=name,
            prio=prio,
            type=type,
            eq_task_ls=eq_task_ls,
            eq_J_T_s=[
                ca.jacobian(eq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))
            ],
            eq_J_T_u=[
                ca.jacobian(eq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))
            ],
            eq_coeff=None
            if eq_task_coeff is None
            else [
                [
                    self.InfList(
                        [e.reshape((-1, 1)) for e in eq_task_coeff[c][j]]
                        if eq_task_coeff[c][j][0] is not None
                        else [None]
                    )
                    for j in range(self.n_robots[c])
                ]
                for c in range(len(self.n_robots))
            ],
            ineq_task_ls=ineq_task_ls,
            ineq_J_T_s=[
                ca.jacobian(ineq_task_ls[c], self._states[c]) for c in range(len(self.n_robots))
            ],
            ineq_J_T_u=[
                ca.jacobian(ineq_task_ls[c], self._inputs[c]) for c in range(len(self.n_robots))
            ],
            ineq_coeff=None
            if ineq_task_coeff is None
            else [
                [
                    self.InfList(
                        [e.reshape((-1, 1)) for e in ineq_task_coeff[c][j]]
                        if ineq_task_coeff[c][j][0] is not None
                        else [None]
                    )
                    for j in range(self.n_robots[c])
                ]
                for c in range(len(self.n_robots))
            ],
            time_index=time_index,
            robot_index=robot_index,
        )

    # ======================================================================== #

    def create_task_bi(
        self,
        name: str,
        prio: int,
        type: TaskType,
        aux: ca.SX,
        mapping: list[ca.SX] | None = None,
        eq_task_ls: ca.SX | None = None,
        eq_task_coeff: list[np.ndarray] | None = None,
        eq_weight: float = 1.0,
        ineq_task_ls: ca.SX | None = None,
        ineq_task_coeff: list[np.ndarray] | None = None,
        ineq_weight: float = 1.0,
        time_index: TaskIndexes = TaskIndexes.All,
        robot_index: list[list[int]] | None = None,
    ):
        """
        Create a HOMPC.Task of type TaskType.Bi.

        Args:
            name (str): _description_
            prio (int): _description_
            type (TaskType): _description_
            aux (ca.SX): _description_
            mapping (list[ca.SX], optional): _description_. Defaults to None.
            eq_task_ls (ca.SX, optional): _description_. Defaults to None.
            eq_task_coeff (list[np.ndarray], optional): _description_. Defaults to None.
            ineq_task_ls (ca.SX, optional): _description_. Defaults to None.
            ineq_task_coeff (list[np.ndarray], optional): _description_. Defaults to None.
            time_index (TaskIndexes, optional): _description_. Defaults to TaskIndexes.All.
        """

        if eq_task_ls is None:
            eq_task_ls = ca.SX.sym('eq', 0)
        if ineq_task_ls is None:
            ineq_task_ls = ca.SX.sym('ineq', 0)

        self._tasks.append(
            self.Task(
                name=name,
                prio=prio,
                type=type,
                eq_task_ls=eq_task_ls,
                eq_J_T_s=None,
                eq_J_T_u=None,
                eq_coeff=eq_task_coeff,
                eq_weight=eq_weight,
                ineq_task_ls=ineq_task_ls,
                ineq_J_T_s=None,
                ineq_J_T_u=None,
                ineq_coeff=ineq_task_coeff,
                ineq_weight=ineq_weight,
                aux_var=aux,
                mapping=mapping,
                time_index=time_index,
                robot_index=robot_index,
            )
        )

    # ! new function

    def update_task_bi(
        self,
        name: str,
        prio: int | None = None,
        type: TaskType | None = None,
        aux: ca.SX | None = None,
        mapping: list[ca.SX] | None = None,
        eq_task_ls: ca.SX | None = None,
        eq_task_coeff: list[np.ndarray] | None = None,
        eq_weight: float = 1.0,
        ineq_task_ls: ca.SX | None = None,
        ineq_task_coeff: list[np.ndarray] | None = None,
        ineq_weight: float = 1.0,
        time_index: TaskIndexes = TaskIndexes.All,
        robot_index: TaskIndexes | None = None,
        pos: int | None = None,
    ):
        for i, t in enumerate(self._tasks):
            if t.name == name:
                if pos is not None and i == pos:
                    id = i
                    break

        if prio is None:
            prio = self._tasks[id].prio
        if type is None:
            type = self._tasks[id].type
        if aux is None:
            aux = self._tasks[id].aux_var
        if mapping is None:
            mapping = self._tasks[id].mapping
        if eq_task_ls is None:
            eq_task_ls = self._tasks[id].eq_task_ls
        if eq_task_coeff is None:
            eq_task_coeff = self._tasks[id].eq_coeff
        if eq_weight is None:
            eq_weight = self._tasks[id].eq_weight
        if ineq_task_ls is None:
            ineq_task_ls = self._tasks[id].ineq_task_ls
        if ineq_task_coeff is None:
            ineq_task_coeff = self._tasks[id].ineq_coeff
        if time_index is None:
            time_index = self._tasks[id].time_index
        if ineq_weight is None:
            ineq_weight = self._tasks[id].ineq_weight
        if robot_index is None:
            robot_index = self._tasks[id].robot_index

        self._tasks[i] = self.Task(
            name=name,
            prio=prio,
            type=type,
            eq_task_ls=eq_task_ls,
            eq_J_T_s=None,
            eq_J_T_u=None,
            eq_coeff=eq_task_coeff,
            eq_weight=eq_weight,
            ineq_task_ls=ineq_task_ls,
            ineq_J_T_s=None,
            ineq_J_T_u=None,
            ineq_coeff=ineq_task_coeff,
            ineq_weight=ineq_weight,
            aux_var=aux,
            mapping=mapping,
            time_index=time_index,
            robot_index=robot_index,
        )

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
            timesteps = (
                list(range(n_c + n_p))
                if t.time_index == TaskIndexes.All
                else [n_c + n_p]
                if t.time_index == TaskIndexes.Last
                else t.time_index
            )

            if t.robot_index is None:
                ne = sum(
                    np.multiply(
                        [t.eq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                        self.n_robots,
                    )
                ) * len(timesteps)

                ni = sum(
                    np.multiply(
                        [t.ineq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                        self.n_robots,
                    )
                ) * len(timesteps)
            else:
                ne = sum(
                    np.multiply(
                        [t.eq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                        [len(t.robot_index[c]) for c in range(len(t.robot_index))],
                    )
                ) * len(timesteps)

                ni = sum(
                    np.multiply(
                        [t.ineq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                        [len(t.robot_index[c]) for c in range(len(t.robot_index))],
                    )
                ) * len(timesteps)

            A = np.zeros((ne, self._get_n_x_opt()))
            b = np.zeros((ne, 1))
            C = np.zeros((ni, self._get_n_x_opt()))
            d = np.zeros((ni, 1))

            ie = 0
            ii = 0
            for c, n_r in enumerate(self.n_robots):
                j_list = range(n_r) if t.robot_index is None else t.robot_index[c]
                # if j_list[0] >= n_r:
                #      j_list = range(n_r)
                for j in j_list:  #! not j_list ma index on j_list-> for j, _ in enumerate(j_list):
                    for k in timesteps:
                        ne = t.eq_J_T_s[c].shape[0]
                        ni = t.ineq_J_T_s[c].shape[0]

                        [
                            A[ie : ie + ne, self._get_idx_state_kp1(c, j, k)],
                            A[ie : ie + ne, self._get_idx_input_k(c, j, k)],
                            b[ie : ie + ne],
                            C[ii : ii + ni, self._get_idx_state_kp1(c, j, k)],
                            C[ii : ii + ni, self._get_idx_input_k(c, j, k)],
                            d[ii : ii + ni],
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
                            A[k * ne : (k + 1) * ne, self._get_idx_state_kp1(c, j, k)],
                            A[k * ne : (k + 1) * ne, self._get_idx_input_k(c, j, k)],
                            b_temp,
                            C[k * ni : (k + 1) * ni, self._get_idx_state_kp1(c, j, k)],
                            C[k * ni : (k + 1) * ni, self._get_idx_input_k(c, j, k)],
                            d_temp,
                        ] = self._helper_create_task_i_matrices(t, c, j, k)

                        b[k * ne : (k + 1) * ne] += b_temp
                        d[k * ni : (k + 1) * ni] += d_temp

        # ==================================================================== #

        elif t.type == TaskType.SameTimeDiff:
            ne = sum(
                np.multiply(
                    [t.eq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                    self.n_robots,
                )
            ) * (n_c + n_p)

            ni = sum(
                np.multiply(
                    [t.ineq_J_T_s[c].shape[0] for c in range(len(self.n_robots))],
                    self.n_robots,
                )
            ) * (n_c + n_p)

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
                            A[ie : ie + ne, self._get_idx_state_kp1(c, j, k)],
                            A[ie : ie + ne, self._get_idx_input_k(c, j, k)],
                            b[ie : ie + ne],
                            C[ii : ii + ni, self._get_idx_state_kp1(c, j, k)],
                            C[ii : ii + ni, self._get_idx_input_k(c, j, k)],
                            d[ii : ii + ni],
                        ] = self._helper_create_task_i_matrices(t, c, j, k)

                        [
                            A[ie : ie + ne, self._get_idx_state_kp1(c, j, k - 1)],
                            A[ie : ie + ne, self._get_idx_input_k(c, j, k - 1)],
                            b[ie : ie + ne],
                            C[ii : ii + ni, self._get_idx_state_kp1(c, j, k - 1)],
                            C[ii : ii + ni, self._get_idx_input_k(c, j, k - 1)],
                            d[ii : ii + ni],
                        ] = [-e for e in self._helper_create_task_i_matrices(t, c, j, k - 1)]

                        if k > self.n_control - 1:
                            ki = self.n_control - 1
                        else:
                            ki = k

                        eq_coeff = 0 if t.eq_coeff is None else t.eq_coeff[c][j][k]

                        b[ie : ie + ne] = (
                            eq_coeff
                            - subs(
                                [t.eq_task_ls[c]],
                                [self._states[c], self._inputs[c]],
                                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
                            )
                            + subs(
                                [t.eq_task_ls[c]],
                                [self._states[c], self._inputs[c]],
                                [self._state_bar[c][j][k], self._input_bar[c][j][ki - 1]],
                            )
                        )

                        ineq_coeff = 0 if t.ineq_coeff is None else t.ineq_coeff[c][j][k]

                        d[ii : ii + ni] = (
                            ineq_coeff
                            - subs(
                                [t.ineq_task_ls[c]],
                                [self._states[c], self._inputs[c]],
                                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
                            )
                            + subs(
                                [t.ineq_task_ls[c]],
                                [self._states[c], self._inputs[c]],
                                [self._state_bar[c][j][k], self._input_bar[c][j][ki - 1]],
                            )
                        )

                        ie += ne
                        ii += ni

        elif t.type == TaskType.Bi:
            if t.eq_coeff is None and t.ineq_coeff is None:
                ne = (int)(binom(sum(self.n_robots), 2) * t.eq_task_ls.size1()) * (n_c + n_p)
                ni = (int)(binom(sum(self.n_robots), 2) * t.ineq_task_ls.size1()) * (n_c + n_p)

                A = np.zeros((ne, self._get_n_x_opt()))
                b = np.zeros((ne, 1))
                C = np.zeros((ni, self._get_n_x_opt()))
                d = np.zeros((ni, 1))

                ie = 0
                ii = 0
                for e_c in set(
                    itertools.combinations([i for i in range(len(self.n_robots))] * 2, 2)
                ):
                    c0, c1 = (e_c[0], e_c[0]) if e_c[0] == e_c[1] else e_c
                    combinations = (
                        set(itertools.combinations([i for i in range(self.n_robots[c0])], 2))
                        if c0 == c1
                        else list(
                            itertools.product(
                                [i for i in range(self.n_robots[c0])],
                                [i for i in range(self.n_robots[c1])],
                            )
                        )
                    )

                    for e_j in combinations:
                        j0, j1 = e_j

                        ne_block = t.eq_task_ls.shape[0]
                        ni_block = t.ineq_task_ls.shape[0]

                        for k in range(n_c + n_p):
                            [
                                A[ie : ie + ne_block, self._get_idx_state_kp1(c0, j0, k)],
                                A[ie : ie + ne_block, self._get_idx_state_kp1(c1, j1, k)],
                                A[ie : ie + ne_block, self._get_idx_input_k(c0, j0, k)],
                                A[ie : ie + ne_block, self._get_idx_input_k(c1, j1, k)],
                                b[ie : ie + ne_block],
                                C[ii : ii + ni_block, self._get_idx_state_kp1(c0, j0, k)],
                                C[ii : ii + ni_block, self._get_idx_state_kp1(c1, j1, k)],
                                C[ii : ii + ni_block, self._get_idx_input_k(c0, j0, k)],
                                C[ii : ii + ni_block, self._get_idx_input_k(c1, j1, k)],
                                d[ii : ii + ni_block],
                            ] = self._helper_create_task_bi_i_matrices(t, c0, j0, c1, j1, k)

                            ie += ne_block
                            ii += ni_block

            else:
                ne = 0 if t.eq_coeff is None else len(t.eq_coeff) * t.eq_task_ls.shape[0]
                ni = 0 if t.ineq_coeff is None else len(t.ineq_coeff) * t.ineq_task_ls.shape[0]

                A = np.zeros((ne, self._get_n_x_opt()))
                b = np.zeros((ne, 1))
                C = np.zeros((ni, self._get_n_x_opt()))
                d = np.zeros((ni, 1))

                ie = 0
                ii = 0
                if t.eq_coeff is not None:
                    for eq_coeff in t.eq_coeff:
                        c0, j0, c1, j1, k, coeff = eq_coeff.get()

                        ne_block = t.eq_task_ls.shape[0]

                        [
                            A[ie : ie + ne_block, self._get_idx_state_kp1(c0, j0, k)],
                            A[ie : ie + ne_block, self._get_idx_state_kp1(c1, j1, k)],
                            A[ie : ie + ne_block, self._get_idx_input_k(c0, j0, k)],
                            A[ie : ie + ne_block, self._get_idx_input_k(c1, j1, k)],
                            b[ie : ie + ne_block],
                        ] = self._helper_create_task_bi_i_matrices(
                            t, c0, j0, c1, j1, k, self.ConstraintType.Eq
                        )

                        b[ie : ie + ne_block] += coeff

                        ie += ne_block

                if t.ineq_coeff is not None:
                    for ineq_coeff in t.ineq_coeff:
                        c0, j0, c1, j1, k, coeff = ineq_coeff.get()

                        ni_block = t.ineq_task_ls.shape[0]

                        [
                            C[ii : ii + ni_block, self._get_idx_state_kp1(c0, j0, k)],
                            C[ii : ii + ni_block, self._get_idx_state_kp1(c1, j1, k)],
                            C[ii : ii + ni_block, self._get_idx_input_k(c0, j0, k)],
                            C[ii : ii + ni_block, self._get_idx_input_k(c1, j1, k)],
                            d[ii : ii + ni_block],
                        ] = self._helper_create_task_bi_i_matrices(
                            t, c0, j0, c1, j1, k, self.ConstraintType.Ineq
                        )

                        d[ii : ii + ni_block] += coeff

                        ii += ni_block

        return A, b, C, d

    # ==================== _helper_create_task_i_matrices ==================== #

    def _helper_create_task_i_matrices(self, t: Task, c: int, j: int, k: int):
        """
        Auxiliary function to create the matrices A, b, C, d.

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
            ] = [
                jacobian(eq_task, state_c0),
                jacobian(eq_task, input_c0),
                eq_task in the linearization point,
                jacobian(ineq_task, state_c0),
                jacobian(ineq_task, input_c0),
                ineq_task in the linearization point,
            ]
        """

        if k > self.n_control - 1:
            ki = self.n_control - 1
        else:
            ki = k

        if t.eq_coeff is None:
            eq_coeff = 0
        elif t.eq_coeff[c][j][k] is None:
            eq_coeff = 0
        else:
            eq_coeff = t.eq_coeff[c][j][k]  #! eq_coeff changed
        # eq_coeff = 0 if t.eq_coeff is None else t.eq_coeff[c][j][k]
        ineq_coeff = 0 if t.ineq_coeff is None else t.ineq_coeff[c][j][k]

        return [
            subs(
                [t.eq_J_T_s[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
            subs(
                [t.eq_J_T_u[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
            eq_coeff
            - subs(
                [t.eq_task_ls[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
            subs(
                [t.ineq_J_T_s[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
            subs(
                [t.ineq_J_T_u[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
            ineq_coeff
            - subs(
                [t.ineq_task_ls[c]],
                [self._states[c], self._inputs[c]],
                [self._state_bar[c][j][k + 1], self._input_bar[c][j][ki]],
            ),
        ]

    # ======================================================================== #

    def _helper_create_task_bi_i_matrices(
        self,
        t: Task,
        c0: int,
        j0: int,
        c1: int,
        j1: int,
        k: int,
        constr_type: ConstraintType = ConstraintType.Both,
    ):
        """
            Auxiliary matrix to create the matrices A, b, C, d

            Args:
                t (Task): task
                c0 (int): first robot class index
                j0 (int): robot number in the class c0
                c1 (int): second robot class index
                j1 (int): robot number in     nx.draw(network_graph)
        plt.show()the class c1
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
                ] = [
                    jacobian(eq_task, state_c0),
                    jacobian(eq_task, state_c1),
                    jacobian(eq_task, input_c0),
                    jacobian(eq_task, input_c1),
                    eq_task in the linearization point,
                    jacobian(ineq_task, state_c0),
                    jacobian(ineq_task, state_c1),
                    jacobian(ineq_task, input_c0),
                    jacobian(ineq_task, input_c1),
                    ineq_task in the linearization point,
                ]
        """

        if k > self.n_control - 1:
            ki = self.n_control - 1
        else:
            ki = k

        def J_f_var(self, task_ls: ca.SX, ci: int, ji: int, derivating_var: ca.SX):
            """Return jacobian(task_ls, derivating_var) computed in x_bar, u_bar."""
            if ci == c0 and ji == j0:
                i = 0
            else:
                i = 1

            return (
                subs(
                    [
                        ca.jacobian(task_ls, t.aux_var[i, :])
                        @ ca.jacobian(t.mapping[ci], derivating_var[ci])
                    ],
                    [
                        self._states[ci],
                        self._inputs[ci],
                        ca.vertcat(t.aux_var[0, :].T),
                        ca.vertcat(t.aux_var[1, :].T),
                    ],
                    [
                        self._state_bar[ci][ji][k + 1],
                        self._input_bar[ci][ji][ki],
                        subs(
                            [t.mapping[c0]],
                            [self._states[c0], self._inputs[c0]],
                            [self._state_bar[c0][j0][k + 1], self._input_bar[c0][j0][ki]],
                        ),
                        subs(
                            [t.mapping[c1]],
                            [self._states[c1], self._inputs[c1]],
                            [self._state_bar[c1][j1][k + 1], self._input_bar[c1][j1][ki]],
                        ),
                    ],
                ),
            )

        def f_in_x_bar_u_bar(self, task: ca.SX):
            """Returns task_ls computed in x_bar, u_bar."""
            return -subs(
                [task],
                [ca.vertcat(t.aux_var[0, :].T), ca.vertcat(t.aux_var[1, :].T)],
                [
                    subs(
                        [t.mapping[c0]],
                        [self._states[c0], self._inputs[c0]],
                        [self._state_bar[c0][j0][k + 1], self._input_bar[c0][j0][ki]],
                    ),
                    subs(
                        [t.mapping[c1]],
                        [self._states[c1], self._inputs[c1]],
                        [self._state_bar[c1][j1][k + 1], self._input_bar[c1][j1][ki]],
                    ),
                ],
            )

        if constr_type == self.ConstraintType.Eq:
            return [
                J_f_var(self, t.eq_task_ls, c0, j0, self._states),
                J_f_var(self, t.eq_task_ls, c1, j1, self._states),
                J_f_var(self, t.eq_task_ls, c0, j0, self._inputs),
                J_f_var(self, t.eq_task_ls, c1, j1, self._inputs),
                f_in_x_bar_u_bar(self, t.eq_task_ls),
            ]
        if constr_type == self.ConstraintType.Ineq:
            return [
                J_f_var(self, t.ineq_task_ls, c0, j0, self._states),
                J_f_var(self, t.ineq_task_ls, c1, j1, self._states),
                J_f_var(self, t.ineq_task_ls, c0, j0, self._inputs),
                J_f_var(self, t.ineq_task_ls, c1, j1, self._inputs),
                f_in_x_bar_u_bar(self, t.ineq_task_ls),
            ]

        return [
            J_f_var(self, t.eq_task_ls, c0, j0, self._states),
            J_f_var(self, t.eq_task_ls, c1, j1, self._states),
            J_f_var(self, t.eq_task_ls, c0, j0, self._inputs),
            J_f_var(self, t.eq_task_ls, c1, j1, self._inputs),
            f_in_x_bar_u_bar(self, t.eq_task_ls),
            J_f_var(self, t.ineq_task_ls, c0, j0, self._states),
            J_f_var(self, t.ineq_task_ls, c1, j1, self._states),
            J_f_var(self, t.ineq_task_ls, c0, j0, self._inputs),
            J_f_var(self, t.ineq_task_ls, c1, j1, self._inputs),
            f_in_x_bar_u_bar(self, t.ineq_task_ls),
        ]

    # ======================================================================== #

    def __call__(
        self,
        state_meas: np.ndarray = None,
        rho_delta: np.ndarray = None,
        inputs: list[np.ndarray] = None,
        id: int = None,
    ) -> np.ndarray:
        start_time = time.time()

        n_c = self._n_control

        self._initialize(state_meas, inputs)

        # ================ Reorder Tasks And Create Matrices ================ #
        if not stack:
            n_tasks = len(self._tasks)

            prio = [x.prio for x in self._tasks]
            prio = [0] + prio

            A = [None] * (1 + n_tasks)
            b = [None] * (1 + n_tasks)
            C = [None] * (1 + n_tasks)
            d = [None] * (1 + n_tasks)

            A[0], b[0] = self._task_dynamics_consistency()

            self.solve_times['Create Problem'] += time.time() - start_time

            self._tasks = sorted(self._tasks, key=lambda x: x.prio)

            for k in range(n_tasks):
                kp = k + 1
                A[kp], b[kp], C[kp], d[kp] = self._create_task_i_matrices(k)
        else:
            self._tasks = sorted(self._tasks, key=lambda x: x.prio)
            prio = [x.prio for x in self._tasks]
            prio = [0] + prio

            n_prio = len(
                {t.prio for t in self._tasks}
            )  # set comprehension to get unique priorities
            A = [None] * (1 + n_prio)
            b = [None] * (1 + n_prio)
            C = [None] * (1 + n_prio)
            d = [None] * (1 + n_prio)

            A[0], b[0] = self._task_dynamics_consistency()

            self.solve_times['Create Problem'] += time.time() - start_time

            p = 1
            for k, t in enumerate(self._tasks):
                if k == 0:  # otherwise self._tasks[k-1] creates problems
                    A[p], b[p], C[p], d[p] = self._create_task_i_matrices(k)
                    p += 1
                    continue

                if t.prio != self._tasks[k - 1].prio:
                    A[p], b[p], C[p], d[p] = self._create_task_i_matrices(k)
                    p += 1
                else:
                    A_temp, b_temp, C_temp, d_temp = self._create_task_i_matrices(k)
                    A[p - 1] = np.vstack((A[p - 1], A_temp))
                    b[p - 1] = np.vstack((b[p - 1], b_temp))
                    C[p - 1] = np.vstack((C[p - 1], C_temp))
                    d[p - 1] = np.vstack((d[p - 1], d_temp))

                    # # reduce tasks
                    # Ab = np.hstack((A[p-1], b[p-1]))
                    # Ab_sym = Matrix(Ab)
                    # basis_rows = Ab_sym.rref()[0]  # reduced row echelon form
                    # A[p-1] = copy.deepcopy(np.array(basis_rows[:,:-1].tolist(), dtype=float))
                    # b[p-1] = copy.deepcopy(np.array(basis_rows[:,-1:].tolist(), dtype=float))

        # =================================================================== #

        # self.solve_times["Create Problem"] += time.time() - start_time

        # hqp = HierarchicalQP(solver=self.solver, hierarchical=self.hierarchical)
        start_time = time.time()
        if self.hierarchical:
            x_star, x_star_p, cost = self.hqp(
                A, b, C, d, rho_delta, self.degree, n_c, prio_list=prio
            )  # slack variable inside cost
        else:
            we = [np.inf] + [t.eq_weight for t in self._tasks]
            wi = [np.inf] + [t.ineq_weight for t in self._tasks]
            x_star, x_star_p = self.hqp(A, b, C, d, rho_delta, self.degree, n_c, we, wi)
        self.solve_times['Solve Problem'] += time.time() - start_time

        u_0 = [
            [
                self._input_bar[c][j][0] + x_star[self._get_idx_input_k(c, j, 0)]
                for j in range(self.n_robots[c])
            ]
            for c in range(len(self.n_robots))
        ]

        """u = [
            [self._input_bar[c][j][k] + x_star[self._get_idx_input_k(c, j, k)]
                    for k in range(n_c)
                for j in range(self.n_robots[c])]
            for c in range(len(self.n_robots))
        ]
        
        s = [
            [self._state_bar[c][j][k] + x_star[self._get_idx_state_kp1(c, j, k)]
                    for k in range(n_c)
                for j in range(self.n_robots[c])]
            for c in range(len(self.n_robots))
        ]"""

        # # prepare vector to share with the neighbours
        # x_neigh = []
        # for j in range(1,self.n_robots[1]):
        #     s_j = [np.reshape(self._state_bar[1][j][k], 3) + x_star[self._get_idx_state_kp1(1, j, k)]
        #             for k in range(n_c)]
        #     u_j = [self._input_bar[1][j][k] + x_star[self._get_idx_input_k(1, j, k)]
        #             for k in range(n_c)]
        #     x_neigh.append((j, [s_j, u_j]))

        y = self._y_extraction(x_star_p, n_c)

        for c, n_r in enumerate(self.n_robots):
            for j in range(n_r):
                for k in range(n_c):
                    self._input_bar[c][j][k] = copy.deepcopy(
                        self._input_bar[c][j][k] + x_star[self._get_idx_input_k(c, j, k)]
                    )

        return u_0, y, cost

    # ======================================================================== #

    def _y_extraction(self, x_star_p, n_c) -> list[np.ndarray]:
        """
        Compose the correct y vector from the variational optimization vector
        ! the consensus vector is x_tilde not x

        """
        y_ordering = self._get_n_x_opt_indexing(n_c)
        p = 0
        priority = len(x_star_p)
        while p < priority:
            for c, n_r in enumerate(self.n_robots):
                for j in range(n_r):
                    for k in range(n_c):
                        # x_star_p[p][self._get_idx_state_kp1(c, j, k)] = copy.deepcopy(
                        #         [self._state_bar[c][j][k].T + x_star_p[p][self._get_idx_state_kp1(c, j, k)]])
                        # x_star_p[p][self._get_idx_input_k(c, j, k)] = copy.deepcopy(
                        #         [self._input_bar[c][j][k].T + x_star_p[p][self._get_idx_input_k(c, j, k)]])

                        x_star_p[p][self._get_idx_state_kp1(c, j, k)] = copy.deepcopy(
                            [x_star_p[p][self._get_idx_state_kp1(c, j, k)]]
                        )
                        # x_star_p[p][self._get_idx_input_k(c, j, k)] = copy.deepcopy(
                        #         [x_star_p[p][self._get_idx_input_k(c, j, k)]])
                        # print(f'k: {copy.deepcopy([self._state_bar[c][j][k].T + x_star_p[p][self._get_idx_state_kp1(c, j, k)]])}')

                x_star_p[p] = x_star_p[p][y_ordering]
                if p < priority:
                    p += 1

        return np.array(x_star_p)

    def _get_n_x_opt_indexing(self, n_c) -> np.ndarray:
        """
        Return the desired order of the elements of each robot inside the optimization vector.
        """
        index = np.array([], dtype=int)
        for c, n_r in enumerate(self.n_robots):
            for j in range(n_r):
                for k in range(n_c):
                    index = np.concatenate(
                        (
                            index,
                            self._get_idx_state_kp1(c, j, k),
                        )
                    )
                for k in range(n_c):
                    index = np.concatenate(
                        (
                            index,
                            self._get_idx_input_k(c, j, k),
                        )
                    )

        return index

    def _get_n_x_opt(self) -> int:
        """
        Return the dimension of the optimization vector n_x_opt.
        n_x_opt = [u_0; u_1; ...; u_{n_c-1}; s_1; s_2; ...; s_{n_c+n_p}]
        """

        n_x = 0
        for i, n_r in enumerate(self.n_robots):
            n_x += n_r * (
                self.n_control * self._n_inputs[i]
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

        return np.arange(temp1 + temp2 + k * n_i, temp1 + temp2 + (k + 1) * n_i)

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
            return np.arange(temp1 + temp2 + temp3 + k * n_s, temp1 + temp2 + temp3 + (k + 1) * n_s)

        return np.arange(temp1 + temp2 + temp3 + k * n_s, temp1 + temp2 + temp3 + (k + 1) * n_s)
