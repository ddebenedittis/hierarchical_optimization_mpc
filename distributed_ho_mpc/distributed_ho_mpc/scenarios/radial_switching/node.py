import copy
import csv

import casadi as ca
import numpy as np
from matplotlib import pyplot as plt

import distributed_ho_mpc.scenarios.radial_switching.settings as st
from distributed_ho_mpc.scenarios.radial_switching.ho_mpc.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    TaskBiCoeff,
    TaskIndexes,
    TaskType,
)
from distributed_ho_mpc.scenarios.radial_switching.ho_mpc.robot_models import (
    RobCont,
    get_omnidirectional_model,
    get_unicycle_model,
)
from distributed_ho_mpc.scenarios.radial_switching.message import (
    MessageReceiver,
    MessageSender,
)


class Node:
    """
    Representing the node and its actions and attributes

    Class method:
    -> transmit_data: create a message with information to share with neighbours
    -> update: pop from local memory buffer received messages and update its state
    -> receive_data: append the received message in a local memory buffer

    """

    def __init__(
        self,
        node_id: int,
        adjacency_vector: np.array,
        model: str,
        dt: float,
        self_tasks: list,
        neigh_tasks: dict,
        goals: np.array,
        n_steps: int,
    ):
        super(Node, self).__init__()

        self.node_id = copy.deepcopy(node_id)  # ID of the node
        self.adjacency_vector = copy.deepcopy(adjacency_vector)  # neighbours
        self.neigh = np.nonzero(adjacency_vector)[0].tolist()  # index of neighbours
        self.degree = len(self.neigh)  # numbers of neighbours

        self.x_neigh = []  # local buffer to store primal variables to share
        self.x_i = []
        self.n_priority = st.n_priority  # number of priorities
        self.n_xi = st.n_control * 5  # dimension of primal variables

        self.cost_history = []  # history of cost function values
        # ======================== Variables updater ======================= #
        self.alpha = st.step_size * np.ones(
            self.n_xi * (self.degree)
        )  # step size for primal and dual variables

        self.a = 2

        self.y_i = np.zeros((self.n_priority, self.n_xi * (self.degree + 1)))
        self.rho_i = np.zeros((2, self.n_priority, self.n_xi * (self.degree)))
        # np.random.rand(2, self.n_priority, self.n_xi*(self.degree))*0       # two values for rho_i and rho_j, n_properties rows, n_xi*(degree) columns
        # p1  [[[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
        # p2  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
        # p3  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...]]
        self.y_j = np.zeros(
            (2, self.n_priority, self.n_xi * (self.degree))
        )  # p1  [[[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
        # p2  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
        # p3  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...]]
        self.rho_j = np.zeros(
            (2, self.n_priority, self.n_xi * (self.degree))
        )  # p1  [[[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
        # p2  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
        # p3  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...]]

        self.sender = MessageSender(
            self.node_id, self.neigh, self.y_i, self.rho_i, self.n_xi, self.n_priority
        )

        self.receiver = MessageReceiver(self.node_id, self.neigh, self.y_j, self.rho_j, self.n_xi)

        """self.filename = f"node_{self.node_id}_data.csv"
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            header = ['Time']
            for j in self.neigh:
                for i in range(self.n_xi):
                    header.append(f'rho_(i{j})_i_p3_{i}')
            for j in self.neigh:
                for i in range(self.n_xi):
                    header.append(f'rho_(i{j})_i_p4_{i}') 
            for j in self.neigh:
                for i in range(self.n_xi):
                    header.append(f'rho_({j}i)_i_p3_{i}')
            for j in self.neigh:
                for i in range(self.n_xi):
                    header.append(f'rho_({j}i)_i_p4_{i}') 
            header.append(f'stateX_{self.node_id}')
            header.append(f'stateY_{self.node_id}')
            for j in self.neigh:
                header.append(f'stateX_{j}')
                header.append(f'stateY_{j}')
            header.append(f'inputX_{self.node_id}')
            header.append(f'inputY_{self.node_id}')
            for j in self.neigh:
                header.append(f'inputX_{j}')
                header.append(f'inputY_{j}') 
            header.append('cost')
            # Write the header
            writer.writerow(header) """
        self.filename = f'node_{self.node_id}_data.csv'
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            header = ['Time']
            for i in range(st.n_nodes):
                if i == self.node_id:
                    continue
                for j in range(self.n_xi):
                    header.append(f'rho_(i{i})_p3_{j}')
                for j in range(self.n_xi):
                    header.append(f'rho_(i{i})_p4_{j}')
                for j in range(self.n_xi):
                    header.append(f'rho_({i}i)_p3_{j}')
                for j in range(self.n_xi):
                    header.append(f'rho_({i}i)_p4_{j}')
            for i in range(st.n_nodes):
                header.append(f'stateX_{i}')
                header.append(f'stateY_{i}')
                header.append(f'inputX_{i}')
                header.append(f'inputY_{i}')
            # header.append('cost')

            writer.writerow(header)

        # ======================== Define The System Model ======================= #

        # Define the state and input variables, and the discrete-time dynamics model.

        self.n_robots = RobCont(omni=self.degree + 1)

        self.dt = copy.deepcopy(dt)  # timestep size

        self.s = RobCont(omni=None, uni=None)  # symbolic state variables
        self.u = RobCont(omni=None, uni=None)
        self.s_kp1 = RobCont(omni=None, uni=None)

        # self.s.omni, self.u.omni, self.s_kp1.omni = get_omnidirectional_model(dt*10)
        self.s.omni, self.u.omni, self.s_kp1.omni = get_unicycle_model(dt * 10)

        self.goals = copy.deepcopy(goals)

        self.n_steps = n_steps
        self.step = 0
        self.tasks = self_tasks
        self.neigh_tasks = neigh_tasks

        # shared variable
        self.s_opt = []
        self.u_opt = []

        self.Z_old = []
        self.Z_neigh = {f'{i}': [[np.eye(30)]] for i in self.neigh}  # np.empty([20,20])

        self.v_max = copy.deepcopy(st.v_max)
        self.v_min = copy.deepcopy(st.v_min)

        self.dist_hist = [[], [], []]
        self.delta_hist = [[], [], [], []]
        self.counter = []

    def index_local_to_global(self, r) -> int:
        """
        Convert the local index of the node to the global index in the adjacency vector.
        """
        return self.robot_idx_global[r]

    def index_global_to_local(self, r) -> int:
        """
        Convert the global index of the node to the local index in the adjacency vector.
        """
        return self.robot_idx_global.index(r)

    # ---------------------------------------------------------------------------- #
    #                                     Task                                     #
    # ---------------------------------------------------------------------------- #
    def Tasks(self) -> None:
        "Define the tasks separately"

        # n_robots = [self.degree+1, 0] # n° of neighbours + self agent
        self.robot_idx_global = [self.node_id] + self.neigh
        self.robot_idx = [self.robot_idx_global.index(r) for r in self.robot_idx_global]

        """self.tasks_creator = TasksCreatorHOMPCMultiRobot(
            self.s.tolist(),
            self.u.tolist(),
            self.s_kp1.tolist(),
            st.dt,
            self.n_robots.tolist(),
        )

        self.task_input_limits = self.tasks_creator.get_task_input_limits()
        self.aux, self.mapping, self.task_formation, self.task_formation_coeff = self.tasks_creator.get_task_formation()

        self.task_pos       = [None for i in range(len(self.goals))]
        self.task_pos_coeff = [None for i in range(len(self.goals))]
        for i, g in enumerate(self.goals):
            self.task_pos[i], self.task_pos_coeff[i] = self.tasks_creator.get_task_pos_ref(
                [[g for n_j in range(self.n_robots.tolist())] for c in range(len(self.n_robots.tolist()))], robot_idx=self.robot_idx
        )
        self.task_input_smooth, self.task_input_smooth_coeffs = self.tasks_creator.get_task_input_smooth()
        self.task_input_limits_coeffs = [
            np.array([0, 0, 0, 0])
        ]
        
        self.task_vel_reference = ca.vertcat(
            (self.s_kp1[0] - self.s[0]) / self.dt - 1,
            (self.s_kp1[1] - self.s[1]) / self.dt - 0
        )
            
        self.task_input_min = self.tasks_creator.get_task_input_min()

        # obstacle_pos = np.array([1, 1])
        # obstacle_size = st.formation_distance
        # self.task_obs_avoidance = self.tasks_creator.get_task_obs_avoidance(
        #                                             obstacle_pos, obstacle_size
        #                                     )"""
        # =========================== Define The Tasks ========================== #

        self.task_input_limits = RobCont(
            omni=ca.vertcat(
                self.u.omni[0] - self.v_max,  # vmax
                -self.u.omni[0] + 0,  # vmin
                self.u.omni[1] - 1.5,  # vmax
                -self.u.omni[1] - 1.5,  # vmin
            )
        )

        self.task_input_min = RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1]))

        # ===========================Go-to-Goal====================================== #
        self.task_pos = [None for i in range(len(self.goals))]
        self.task_pos_coeff = [None for i in range(len(self.goals))]
        for i, g in enumerate(self.goals):
            self.task_pos[i] = RobCont(omni=ca.vertcat(self.s_kp1.omni[0], self.s_kp1.omni[1]))
            self.task_pos_coeff[i] = RobCont(
                omni=[[g] for _ in range(self.n_robots.omni)],
            )

        # ========================Formation============================================ #
        if 0:
            self.aux = ca.SX.sym('aux', 2, 2)
            self.mapping = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
            self.task_formation = ca.vertcat(
                (self.aux[0, 0] - self.aux[1, 0]) ** 2 + (self.aux[0, 1] - self.aux[1, 1]) ** 2 - 0,
            )
            if self.node_id == 0:
                self.task_formation_coeff = [
                    TaskBiCoeff(0, 1, 0, 0, 0, 3**2),
                    TaskBiCoeff(0, 2, 0, 3, 0, 3**2),
                    TaskBiCoeff(0, 3, 0, 4, 0, 3**2),
                ]
            elif self.node_id == 1:
                self.task_formation_coeff = [
                    TaskBiCoeff(0, 0, 0, 1, 0, 3**2),
                    TaskBiCoeff(0, 2, 0, 3, 0, 3**2),
                    TaskBiCoeff(0, 3, 0, 4, 0, 3**2),
                ]
            elif self.node_id == 2:
                self.task_formation_coeff = [
                    TaskBiCoeff(0, 0, 0, 2, 0, 3**2),
                    TaskBiCoeff(0, 1, 0, 2, 0, 3**2),
                    TaskBiCoeff(0, 3, 0, 4, 0, 3**2),
                ]
            elif self.node_id == 3:
                self.task_formation_coeff = [
                    TaskBiCoeff(0, 1, 0, 2, 0, 3**2),
                    TaskBiCoeff(0, 0, 0, 3, 0, 3**2),
                    TaskBiCoeff(0, 0, 0, 4, 0, 3**2),
                ]
            elif self.node_id == 4:
                self.task_formation_coeff = [
                    TaskBiCoeff(0, 1, 0, 2, 0, 3**2),
                    TaskBiCoeff(0, 4, 0, 3, 0, 3**2),
                    TaskBiCoeff(0, 0, 0, 4, 0, 3**2),
                ]

        self.mapping = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))

        # =====================Collision Avoidance=================================== #
        self.threshold = 2
        self.aux_avoid_collision = ca.SX.sym('aux', 2, 2)
        self.mapping_avoid_collision = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
        self.task_avoid_collision = ca.vertcat(
            -((self.aux_avoid_collision[0, 0] - self.aux_avoid_collision[1, 0]) ** 2)
            - (self.aux_avoid_collision[0, 1] - self.aux_avoid_collision[1, 1]) ** 2,
        )
        self.task_avoid_collision_coeff = [
            TaskBiCoeff(0, 0, 0, j, 0, -(self.threshold**2)) for j in self.robot_idx[1:]
        ]
        for p, j in enumerate(self.robot_idx[1:]):
            for pp in self.robot_idx[p + 1 :]:
                self.task_avoid_collision_coeff.append(
                    TaskBiCoeff(0, j, 0, pp, 0, -(self.threshold**2))
                )

        # =====================Obstacle Avoidance===================================== #
        self.obstacle_pos = np.array([2, 2])
        self.obstacle_size = 3
        self.task_obs_avoidance = [
            ca.vertcat(
                -((self.s.omni[0] - self.obstacle_pos[0]) ** 2)
                - (self.s.omni[1] - self.obstacle_pos[1]) ** 2
                + self.obstacle_size**2
            )
        ]

    def task_formation_method(self, agents, distance):
        aux = ca.SX.sym('aux', 2, 2)
        # mapping = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
        task_formation = ca.vertcat(
            (aux[0, 0] - aux[1, 0]) ** 2 + (aux[0, 1] - aux[1, 1]) ** 2 - 0,
        )
        agents_0 = self.index_global_to_local(agents[0][0])  # convert global index to local index
        agents_1 = self.index_global_to_local(agents[0][1])  # convert global index to local index
        task_formation_coeff = [
            TaskBiCoeff(0, agents_0, 0, agents_1, 0, distance**2),
        ]
        formation_index = [[agents_0, agents_1]]
        # formation_index = [[agents_1]]
        return aux, self.mapping, task_formation, task_formation_coeff, formation_index

    # ---------------------------------------------------------------------------- #
    #                                      MPC                                     #
    # ---------------------------------------------------------------------------- #
    def MPC(self) -> None:
        self.hompc = HOMPCMultiRobot(
            self.s.tolist(),
            self.u.tolist(),
            self.s_kp1.tolist(),
            self.n_robots.tolist(),
            self.degree,
        )
        self.hompc.n_control = st.n_control
        self.hompc.n_pred = st.n_pred

        # ======================================================================== #

        for task in self.tasks:
            if task['name'] == 'input_limits':
                self.hompc.create_task(
                    name='input_limits',
                    prio=task['prio'],
                    type=TaskType.Same,
                    ineq_task_ls=self.task_input_limits.tolist(),
                    robot_index=[self.robot_idx],
                    # ineq_task_coeff= self.task_input_limits_coeffs
                )
            elif task['name'] == 'position':
                self.hompc.create_task(
                    name='position',
                    prio=task['prio'],
                    type=TaskType.Same,
                    eq_task_ls=self.task_pos[task['goal_index']].tolist(),
                    eq_task_coeff=self.task_pos_coeff[task['goal_index']].tolist(),
                    time_index=TaskIndexes.All,
                    robot_index=[[0]],
                )
            elif task['name'] == 'input_minimization':
                self.hompc.create_task(
                    name='input_minimization',
                    prio=task['prio'],
                    type=TaskType.Same,
                    eq_task_ls=self.task_input_min.tolist(),
                    robot_index=[self.robot_idx],
                )
            elif task['name'] == 'input_smooth':
                self.hompc.create_task(
                    name='input_smooth',
                    prio=task['prio'],
                    type=TaskType.SameTimeDiff,
                    ineq_task_ls=RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1])).tolist(),
                    # ineq_task_coeff = np.array([0,0,0,0]),
                    robot_index=[self.robot_idx],
                )
            elif task['name'] == 'formation':
                aux, mapping, task_formation, task_formation_coeff, f_robot_idx = (
                    self.task_formation_method(task['agents'], task['distance'])
                )
                self.hompc.create_task_bi(
                    name='formation',
                    prio=task['prio'],
                    type=TaskType.Bi,
                    aux=aux,
                    mapping=self.mapping.tolist(),
                    eq_task_ls=task_formation,
                    eq_task_coeff=task_formation_coeff,
                    robot_index=f_robot_idx,
                )
            elif task['name'] == 'collision_avoidance' and self.degree > 0:
                self.hompc.create_task_bi(
                    name='collision',
                    prio=task['prio'],
                    type=TaskType.Bi,
                    aux=self.aux_avoid_collision,
                    mapping=self.mapping_avoid_collision.tolist(),
                    ineq_task_ls=self.task_avoid_collision,
                    ineq_task_coeff=self.task_avoid_collision_coeff,
                    robot_index=[self.robot_idx[1:]],
                )
            elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                    name='obstacle_avoidance',
                    prio=task['prio'],
                    type=TaskType.Same,
                    ineq_task_ls=self.task_obs_avoidance,
                )
        for neigh in self.neigh_tasks:
            self.create_neigh_tasks(neigh)

        # ======================================================================== #

        if self.node_id == 0:
            self.s = RobCont(
                omni=[np.array([-4, 4, 0.1]) for _ in range(self.n_robots.omni)],
            )
        elif self.node_id == 1:
            self.s = RobCont(omni=[np.array([3.5, 4, -2.1]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 2:
            self.s = RobCont(omni=[np.array([3.5, -4, 2.1]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 3:
            self.s = RobCont(omni=[np.array([-4, -4, 0.75]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 4:
            self.s = RobCont(omni=[np.array([-5, -2, 0.25]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 5:
            self.s = RobCont(omni=[np.array([5, 2, 3]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 6:
            self.s = RobCont(omni=[np.array([-5, 2, 0.25]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 7:
            self.s = RobCont(omni=[np.array([5, -2, 3]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 8:
            self.s = RobCont(omni=[np.array([-7, 0, 0.25]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 9:
            self.s = RobCont(omni=[np.array([7, 0, 3]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 10:
            self.s = RobCont(omni=[np.array([0, -5, 2.1]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 11:
            self.s = RobCont(omni=[np.array([0, 5, -2.1]) for _ in range(self.n_robots.omni)])
        else:
            raise ValueError('Missing agent init on s')

        self.s_history = [None for _ in range(self.n_steps)]
        self.s_history_p = [None for _ in range(self.n_steps)]
        self.s_init = copy.deepcopy(self.s)
        return

    # ---------------------------------------------------------------------------- #
    #                                Node's Methods                                #
    # ---------------------------------------------------------------------------- #

    def reorder_s_init(self, state_meas: list[float]):
        for j, s_j in enumerate(state_meas):
            if j in self.robot_idx_global:
                self.s_init.omni[self.index_global_to_local(j)] = copy.deepcopy(
                    s_j
                )  # TODO manage eterogeneous robots

        # update position of other robots (not neigh) seen as obstacles
        # self.obstacle_pos = state_meas[2]
        # self.task_obs_avoidance = [
        #     ca.vertcat(- (self.s.omni[0][0] - self.obstacle_pos[0])**2 - (self.s.omni[0][1] - self.obstacle_pos[1])**2 + self.obstacle_size**2)
        # ]

        # if self.node_id == 1 or self.node_id == 2 or self.node_id == 0:
        #         self.hompc.update_task(
        #                     name = "obstacle_avoidance",
        #                     ineq_task_ls = self.task_obs_avoidance,
        #                     # robot_index = cov_rob_idx,
        #                 )

    def receive_data(self, message) -> None:
        "Append the received information in a local buffer"

        self.receiver.receive_message(message)

    def transmit_data(self, receiver_id: int, update: str):
        "Create a message with primal variables state and the neighbours to share with"

        return self.sender.send_message(receiver_id, update)

    def update(self):
        """Pop from local buffer the received dual variables of neighbours and minimize primal function"""

        if self.step != 0:
            self.rho_j = self.receiver.process_messages('D')

        if self.step < self.n_steps:
            print(self.step)
            rho_delta = self.rho_i - self.rho_j  #! to be controlled

            self.u_star, self.y, cost = self.hompc(copy.deepcopy(self.s.tolist()), rho_delta)
            self.sender.y = copy.deepcopy(self.y)  # update copy of the states to share

            self.y_i = copy.deepcopy(self.y)

            self.cost_history.append(cost)
            # put in message u and s
            if self.step % self.a == 0:
                self.s = self.evolve(
                    copy.deepcopy(self.s_init), RobCont(omni=self.u_star[0]), self.dt
                )
                # self.a = self.a * 2
                self.counter.append(self.step)
            else:
                self.s = self.evolve(self.s, RobCont(omni=self.u_star[0]), self.dt)

            if st.inner_plot:
                self.s_ = self.evolve(self.s, RobCont(omni=self.u_star[0]), self.dt)

                for i in range(len(self.s_.omni)):
                    self.delta_hist[i].append(np.linalg.norm(self.s_.omni[i] - self.s.omni[i]))
                    if i != 0:
                        self.dist_hist[i - 1].append(
                            np.linalg.norm(self.s_.omni[0] - self.s.omni[i])
                        )
                """if self.step == 900 and (self.node_id==0):# or self.node_id==1):
                        self.goals = [
                            np.array([-6, -7]),
                            np.array([-4, -5]),
                        ]
                        for i, g in enumerate(self.goals):
                            self.task_pos[i] = RobCont(omni=ca.vertcat(self.s_kp1.omni[0], self.s_kp1.omni[1]))
                            self.task_pos_coeff[i] = RobCont(
                                omni=[[g] for _ in range(self.n_robots.omni)],
                            )
                    
                        self.hompc.update_task(
                            name = "position",
                            eq_task_coeff = self.task_pos_coeff[0].tolist(),
                            # robot_index = cov_rob_idx,
                        )"""

                if self.step == st.n_steps - 1:
                    plt.figure(figsize=(10, 6))
                    plt.suptitle(f'Node_{self.node_id}  and Delta')

                    for i, d in enumerate(self.dist_hist):
                        plt.subplot(211)
                        plt.title('intra-distances')
                        plt.plot(d, label=f'coppia 0-{i + 1}')
                        plt.grid()
                        plt.legend()
                    for i, d in enumerate(self.delta_hist):
                        plt.subplot(212)
                        plt.title('error on state')
                        plt.plot(d, label=f'Delta {i + 1}', linestyle='--')
                    plt.legend()
                    plt.show()

            print(f's:\t{self.s.tolist()}\nu:\t{self.u_star}\n')

            self.s_history[self.step] = copy.deepcopy(self.s.tolist())
            self.s_history_p[self.step] = copy.deepcopy([self.s.omni[0]])
            self.step += 1

        return

    def dual_update(self):
        """Update the dual variables rho_i and rho_j using the received messages from neighbours"""

        # if self.step > 0:
        #     self.save_data()

        self.y_j = self.receiver.process_messages('P')

        # linear update of rho_i
        self.rho_i[0, :, :] += self.alpha * (
            np.tile(self.y_i[:, 0 : self.n_xi], self.degree) - self.y_j[0, :, :]
        )
        self.rho_i[1, :, :] += self.alpha * (self.y_i[:, self.n_xi :] - self.y_j[1, :, :])

        self.sender.rho = copy.deepcopy(self.rho_i)  # update copy of the states to share

    def evolve(self, s: list[list[float]], u_star: list[list[float]], dt: float):
        """Update the state of the system using the control input u_star and the time step dt"""

        n_intervals = 10
        for j, _ in enumerate(s.omni):
            for _ in range(n_intervals):
                s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
                    [
                        u_star.omni[j][0] * np.cos(s.omni[j][2]),
                        u_star.omni[j][0] * np.sin(s.omni[j][2]),
                        u_star.omni[j][1],
                    ]
                )
        # for j, _ in enumerate(s.omni):
        #     for _ in range(n_intervals):
        #         s.omni[j] = s.omni[j] + dt / n_intervals * np.array([
        #             u_star.omni[j][0],
        #             u_star.omni[j][1],
        #         ])

        return s

    def plot_dual(self):
        return self.rho_i, self.neigh

    def save_data(self):
        # TODO: partizionare vettori e mettere none
        """if not st.save_data or self.step <= 20:
            return
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the data
            row = [self.step]
            row.extend(self.rho_i[0, 0, :])
            row.extend(self.rho_i[0, 1, :])
            row.extend(self.rho_j[0, 0, :])
            row.extend(self.rho_j[0, 1, :])
            for s in self.s.tolist():
                for ss in s:
                    row.extend(ss)
            for u in self.u_star[0]:
                row.extend(list(u))
            row.append(self.cost_history[-1])


            writer.writerow(row)"""
        if not st.save_data:
            return
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [self.step]
            for i in range(st.n_nodes):
                if i == self.node_id:
                    continue
                if i in self.neigh:
                    ii = self.neigh.index(i)
                    row.extend(self.rho_i[0, 0, (ii * self.n_xi) : (ii + 1) * self.n_xi])
                    row.extend(self.rho_i[0, 1, (ii * self.n_xi) : (ii + 1) * self.n_xi])
                    row.extend(self.rho_j[0, 0, (ii * self.n_xi) : (ii + 1) * self.n_xi])
                    row.extend(self.rho_j[0, 1, (ii * self.n_xi) : (ii + 1) * self.n_xi])
                else:
                    row.extend([None] * (self.n_xi * 4))
            for i in range(st.n_nodes):
                if i in self.robot_idx_global:
                    ii = self.index_global_to_local(i)
                    row.extend(self.s.omni[ii])
                    row.extend(self.u_star[0][ii])
                else:
                    row.extend([None] * 4)
            # row.append(self.cost_history[-1])

            writer.writerow(row)

    def create_neigh_tasks(self, neigh):
        """
        Create the tasks for the HOMPCMultiRobot instance
        """
        is_formation_with_neigh = lambda agents, neigh: all(
            item in neigh for item in agents
        )  # check if neighour's formation is with current agent's neighbour

        robot_idx = None
        for i in self.neigh:
            if neigh == f'agent_{i}':
                robot_idx = self.robot_idx_global.index(i)
                break
            # if f'agent_{i}' in neigh:
            #     robot_idx = self.robot_idx_global.index(i)
            #     break
        if robot_idx is None:
            raise ValueError(f'Could not find robot index for neighbor {neigh.key}')
        for task in self.neigh_tasks[neigh]:
            if task['name'] == 'position':
                self.hompc.create_task(
                    name='position',
                    prio=task['prio'],
                    type=TaskType.Same,
                    eq_task_ls=self.task_pos[task['goal_index']].tolist(),
                    eq_task_coeff=self.task_pos_coeff[task['goal_index']].tolist(),
                    time_index=TaskIndexes.All,
                    robot_index=[[robot_idx]],
                )
            elif task['name'] == 'formation':
                for t in task['agents']:
                    if is_formation_with_neigh(t, self.robot_idx_global):
                        (
                            aux,
                            mapping,
                            task_formation,
                            task_formation_coeff,
                            f_robot_idx,
                        ) = self.task_formation_method(task['agents'], task['distance'])
                        self.hompc.create_task_bi(
                            name='formation',
                            prio=task['prio'],
                            type=TaskType.Bi,
                            aux=aux,
                            mapping=self.mapping.tolist(),
                            eq_task_ls=task_formation,
                            eq_task_coeff=task_formation_coeff,
                            robot_index=f_robot_idx,
                        )
            # elif task['name'] == 'collision_avoidance':
            #     self.hompc.create_task_bi(
            #         name = "collision", prio = task['prio'],
            #         type = TaskType.Bi,
            #         aux = self.aux_avoid_collision,
            #         mapping = self.mapping_avoid_collision.tolist(),
            #         ineq_task_ls= self.task_avoid_collision,
            #         ineq_task_coeff= self.task_avoid_collision_coeff,
            #         robot_index= [self.robot_idx[1:]]
            #     )
            elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                    name='obstacle_avoidance',
                    prio=task['prio'],
                    type=TaskType.Same,
                    ineq_task_ls=self.task_obs_avoidance,
                )

    def create_connection(
        self, adjacency_vector: np.array, neigh_task: dict, state_meas: list[float]
    ):
        """
        .
        """
        # TODO: save data to plot

        added_robot = len(np.nonzero(adjacency_vector)[0].tolist()) + 1 - self.n_robots.omni
        if added_robot > 0:
            neigh = np.nonzero(adjacency_vector)[0].tolist()
            self.adjacency_vector = adjacency_vector
            self.degree = len(neigh)
            self.n_robots = RobCont(omni=self.degree + 1)

            # self.robot_idx_global = [self.node_id] + self.neigh
            new_neigh = list(set(neigh) - set(self.neigh))  # index of the new connected neighbours
            self.robot_idx_global.extend(new_neigh)
            self.neigh.extend(new_neigh)
            self.robot_idx = [self.robot_idx_global.index(r) for r in self.robot_idx_global]
            neigh_local_idx = [self.index_global_to_local(r) for r in new_neigh]

            # expand the consensus variables
            self.y_i = np.pad(
                self.y_i,
                ((0, 0), (0, self.n_xi * (self.degree + 1) - self.y_i.shape[1])),
                mode='constant',
                constant_values=0,
            )
            self.rho_i = np.pad(
                self.rho_i,
                ((0, 0), (0, 0), (0, self.n_xi * (self.degree) - self.rho_i.shape[2])),
                mode='constant',
                constant_values=0,
            )
            self.rho_j = np.pad(
                self.rho_j,
                ((0, 0), (0, 0), (0, self.n_xi * (self.degree) - self.rho_j.shape[2])),
                mode='constant',
                constant_values=0,
            )
            self.y_j = np.pad(
                self.y_j,
                ((0, 0), (0, 0), (0, self.n_xi * (self.degree) - self.y_j.shape[2])),
                mode='constant',
                constant_values=0,
            )
            # self.y_i = np.zeros((self.n_priority, self.n_xi*(self.degree+1)))
            # self.rho_i = np.zeros((2, self.n_priority, self.n_xi*(self.degree)))
            # np.random.rand(2, self.n_priority, self.n_xi*(self.degree))*0       # two values for rho_i and rho_j, n_properties rows, n_xi*(degree) columns
            # p1  [[[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
            # p2  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
            # p3  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...]]
            # self.y_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree)))   # p1  [[[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
            # p2  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
            # p3  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...]]
            # self.rho_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) # p1  [[[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
            # p2  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
            # p3  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...]]
            self.alpha = st.step_size * np.ones(self.n_xi * (self.degree))

            self.hompc.degree = self.degree

            self.hompc.add_robots([added_robot], state_meas)

            self.s.expand(state_meas, 'omni')  # expand the state of the robot to be added
            self.s_init.expand(state_meas, 'omni')  # expand the state of the robot to be added

            self.neigh_tasks.update(neigh_task)  # expand dictionary with neighbour tasks

            for neigh in neigh_task:
                self.create_neigh_tasks(neigh)

            for jj in self.robot_idx[:-1]:  # for each robot except the last one
                self.task_avoid_collision_coeff.append(
                    TaskBiCoeff(0, jj, 0, self.robot_idx[-1], 0, -(self.threshold**2))
                )

            if self.degree == 1:
                self.hompc.create_task_bi(
                    name='collision',
                    prio=3,
                    type=TaskType.Bi,
                    aux=self.aux_avoid_collision,
                    mapping=self.mapping_avoid_collision.tolist(),
                    ineq_task_ls=self.task_avoid_collision,
                    ineq_task_coeff=self.task_avoid_collision_coeff,
                    robot_index=[self.robot_idx[1:]],
                )
            else:
                n = [p for p, v in enumerate(self.hompc._tasks) if v.name == 'collision']

                self.hompc.update_task_bi(
                    name='collision',
                    prio=3,
                    type=TaskType.Bi,
                    # aux = self.aux_avoid_collision,
                    # mapping = self.mapping_avoid_collision.tolist(),
                    # ineq_task_ls= self.task_avoid_collision,
                    ineq_task_coeff=self.task_avoid_collision_coeff,
                    robot_index=[self.robot_idx[1:]],
                    pos=n[0],
                )
                # aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
                #                 [[0,1]], 4
                #         )
                # self.hompc.create_task_bi(
                #     name = "formation", prio = 4,
                #     type = TaskType.Bi,
                #     aux = aux,
                #     mapping = mapping.tolist(),
                #     eq_task_ls = task_formation,
                #     eq_task_coeff = task_formation_coeff,
                # )

            # aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
            #         [[0,1]], 3
            #     )
            # self.hompc.create_task_bi(
            #     name = "formation", prio = 3,
            #     type = TaskType.Bi,
            #     aux = aux,
            #     mapping = self.mapping.tolist(),
            #     eq_task_ls = task_formation,
            #     eq_task_coeff = task_formation_coeff,
            # )

            self.hompc.update_task(name='input_limits', prio=1, robot_index=[self.robot_idx])
            self.hompc.update_task(name='input_smooth', prio=2, robot_index=[self.robot_idx])
            self.sender.update(self.neigh, self.y_i, self.rho_i)
            self.receiver.update(self.neigh, self.y_j, self.rho_j)

            # self.filename = f"node_{self.node_id}_data.csv"
            # with open(self.filename, mode='w', newline='') as file:
            #     writer = csv.writer(file)
            #     # Write the header
            #     header = ['Time']
            #     for j in self.neigh:
            #         for i in range(self.n_xi):
            #             header.append(f'rho_(i{j})_i_p3_{i}')
            #     for j in self.neigh:
            #         for i in range(self.n_xi):
            #             header.append(f'rho_(i{j})_i_p4_{i}')
            #     for j in self.neigh:
            #         for i in range(self.n_xi):
            #             header.append(f'rho_({j}i)_i_p3_{i}')
            #     for j in self.neigh:
            #         for i in range(self.n_xi):
            #             header.append(f'rho_({j}i)_i_p4_{i}')
            #     header.append(f'stateX_{self.node_id}')
            #     header.append(f'stateY_{self.node_id}')
            #     for j in self.neigh:
            #         header.append(f'stateX_{j}')
            #         header.append(f'stateY_{j}')
            #     header.append(f'inputX_{self.node_id}')
            #     header.append(f'inputY_{self.node_id}')
            #     for j in self.neigh:
            #         header.append(f'inputX_{j}')
            #         header.append(f'inputY_{j}')
            #     header.append('cost')
            #     # Write the header
            #     writer.writerow(header)

    def remove_connection(self, adjacency_vector: np.array, neigh_tasks: str, neigh_id: int):
        """
        .
        """
        # TODO: save data to plot

        id_to_remove = self.index_global_to_local(neigh_id)

        self.neigh_tasks.pop(neigh_tasks)

        self.hompc.remove_robots([[id_to_remove]])

        self.s.reduce(id_to_remove, 'omni')  # remove the state of the robot to be removed
        self.s_init.reduce(id_to_remove, 'omni')  # remove the state of the robot to be removed

        rho_idx = list(self.neigh).index(neigh_id)

        # remove element from consensus variables
        self.y_i = np.delete(
            self.y_i,
            np.s_[(id_to_remove * self.n_xi) : (id_to_remove + 1) * self.n_xi],
            1,
        )
        self.rho_i = np.delete(
            self.rho_i, np.s_[(rho_idx * self.n_xi) : (rho_idx + 1) * self.n_xi], 2
        )
        self.rho_j = np.delete(
            self.rho_j, np.s_[(rho_idx * self.n_xi) : (rho_idx + 1) * self.n_xi], 2
        )
        self.y_j = np.delete(self.y_j, np.s_[(rho_idx * self.n_xi) : (rho_idx + 1) * self.n_xi], 2)

        # adjust dimension of variables
        self.neigh.pop(rho_idx)
        self.adjacency_vector = copy.deepcopy(adjacency_vector)
        self.degree = len(self.neigh)
        self.n_robots = RobCont(omni=self.degree + 1)
        robot_idx_global_old = copy.deepcopy(self.robot_idx_global)
        robot_idx_old = copy.deepcopy(self.robot_idx)
        self.robot_idx_global = [self.node_id] + self.neigh
        self.robot_idx = [self.robot_idx_global.index(r) for r in self.robot_idx_global]

        self.alpha = st.step_size * np.ones(self.n_xi * (self.degree))
        self.hompc.degree = copy.deepcopy(self.degree)

        # remove tasks related to the removed robot
        # index = [p for p, task in enumerate(self.hompc._tasks) if id_to_remove not in task.robot_index[0]]
        # self.hompc._tasks = self.hompc._tasks[index]
        if self.degree == 0:
            self.hompc._tasks[:] = [
                task
                for task in self.hompc._tasks
                if id_to_remove not in task.robot_index[0] or task.prio < 3
            ]
        else:
            self.hompc._tasks[:] = [
                task
                for task in self.hompc._tasks
                if id_to_remove not in task.robot_index[0]
                or task.prio < 3
                or task.name == 'collision'
            ]

        self.hompc.update_task(name='input_limits', prio=1, robot_index=[self.robot_idx])
        self.hompc.update_task(name='input_smooth', prio=2, robot_index=[self.robot_idx])

        for n, task in enumerate(self.hompc._tasks):
            if task.type == TaskType.Bi and task.prio > 2:
                if task.name == 'formation':
                    c0, j0, c1, j1, k, coeff = task.eq_coeff[0].get()

                    aux, mapping, task_formation, task_formation_coeff, f_robot_idx = (
                        self.task_formation_method(
                            [[robot_idx_global_old[j0], robot_idx_global_old[j1]]],
                            np.sqrt(coeff),
                        )
                    )
                    self.hompc.update_task_bi(
                        name=task.name,
                        prio=task.prio,
                        aux=aux,
                        mapping=self.mapping.tolist(),
                        robot_index=f_robot_idx,
                        eq_task_ls=task_formation,
                        eq_task_coeff=task_formation_coeff,
                        pos=n,
                    )
                elif task.name == 'collision':
                    # id = robot_idx_global_old[task.robot_index[0][0]]
                    # id = self.robot_idx_global.index(id)

                    self.task_avoid_collision_coeff = [
                        TaskBiCoeff(0, 0, 0, j, 0, -(self.threshold**2)) for j in self.robot_idx[1:]
                    ]
                    for p, j in enumerate(self.robot_idx[1:]):
                        for pp in self.robot_idx[p + 1 :]:
                            self.task_avoid_collision_coeff.append(
                                TaskBiCoeff(0, j, 0, pp, 0, -(self.threshold**2))
                            )

                    self.hompc.update_task_bi(
                        name=task.name,
                        prio=task.prio,
                        robot_index=[self.robot_idx[1:]],
                        ineq_task_coeff=self.task_avoid_collision_coeff,
                        pos=n,
                    )

            elif task.prio > 2:
                # self.task_pos_coeff = [None for i in range(len(self.goals))]
                # for i, g in enumerate(self.goals):
                #     self.task_pos_coeff[i] = RobCont(
                #         omni=[[g] for _ in range(self.n_robots.omni)],
                #     )
                while len(task.eq_coeff[0]) < self.n_robots.omni:
                    task.eq_coeff[0].append([None])

                id = robot_idx_global_old[task.robot_index[0][0]]
                id = self.robot_idx_global.index(id)
                self.hompc.update_task(
                    name=task.name,
                    prio=task.prio,
                    robot_index=[[id]],
                    # eq_task_coeff = self.task_pos_coeff[task['goal_index']].tolist(),
                    pos=n,
                )

        self.sender.update(self.neigh, self.y_i, self.rho_i)
        self.receiver.update(self.neigh, self.y_j, self.rho_j)
