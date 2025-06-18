import copy
import time

import numpy as np
import casadi as ca

import csv
from datetime import datetime, timedelta

from hierarchical_optimization_mpc.auxiliary.evolve import evolve
from hierarchical_optimization_mpc.utils.robot_models import get_unicycle_model, get_omnidirectional_model, RobCont

from ho_mpc.ho_mpc import HOMPC
from ho_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskIndexes, TaskType, TaskBiCoeff
from ho_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot 
from distributed_ho_mpc.message import Message, MessageSender, MessageReceiver
import settings as st


class Node():
    """
    Representing the node and its actions and attributes

    Class method:
    -> transmit_data: create a message with information to share with neighbours
    -> update: pop from local memory buffer received messages and update its state
    -> receive_data: append the received message in a local memory buffer
    
    """

    def __init__(self, node_id: int, adjacency_vector: np.array, model: str, dt: float, self_tasks: list, neigh_tasks:dict, goals: np.array, n_steps: int):

        super(Node, self).__init__()

        self.node_id = node_id  # ID of the node
        self.adjacency_vector = adjacency_vector # neighbours
        self.neigh = np.nonzero(adjacency_vector)[0].tolist() # index of neighbours
        self.degree = len(self.neigh) # numbers of neighbours
         
        self.x_neigh = [] # local buffer to store primal variables to share
        self.x_i = [] 
        self.n_priority = st.n_priority # number of priorities
        self.n_xi = st.n_control * 4 # dimension of primal variables

        self.cost_history = [] # history of cost function values
        # ======================== Variables updater ======================= #
        self.alpha = st.step_size * np.ones(self.n_xi * (self.degree)) # step size for primal and dual variables
        
        self.a = 20
        
        self.y_i = np.zeros((self.n_priority, self.n_xi*(self.degree+1)))
        self.rho_i = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) 
        #np.random.rand(2, self.n_priority, self.n_xi*(self.degree))*0       # two values for rho_i and rho_j, n_properties rows, n_xi*(degree) columns
                                                                             # p1  [[[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p2  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p3  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...]]
        self.y_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree)))   # p1  [[[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p2  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p3  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...]]
        self.rho_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) # p1  [[[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
                                                                             # p2  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
                                                                             # p3  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...]]
        
        
        self.sender = MessageSender(
            self.node_id,
            self.neigh,
            self.y_i,
            self.rho_i,
            self.n_xi,
            self.n_priority
        )
        
        self.receiver = MessageReceiver(
            self.node_id,
            self.neigh,
            self.y_j,
            self.rho_j,
            self.n_xi
        )
        
        self.filename = f"node_{self.node_id}_data.csv"
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
            writer.writerow(header)      
        
        # ======================== Define The System Model ======================= #
    
        # Define the state and input variables, and the discrete-time dynamics model.
        
        self.n_robots = RobCont(omni=self.degree + 1)
                
        self.dt = dt       # timestep size
        
        # self.s = [None for _ in range(2)]
        # self.u = [None for _ in range(2)]
        # self.s_kp1 = [None for _ in range(2)]
        self.s = RobCont(omni = None)
        self.u = RobCont(omni = None)
        self.s_kp1 = RobCont(omni = None)
    
        self.s.omni, self.u.omni, self.s_kp1.omni = get_omnidirectional_model(dt*10)
        
        self.goals = goals

        #self.s[0], self.u[0], self.s_kp1[0] = get_unicycle_model(self.dt)
        #self.s[1], self.u[1], self.s_kp1[1] = get_omnidirectional_model(self.dt)

        #self.s, self.u, self.s_kp1 = model
        self.n_steps = n_steps
        self.step = 0
        self.tasks = self_tasks
        self.neigh_tasks = neigh_tasks
     
        # shared variable
        self.s_opt = []
        self.u_opt = []

        #self.u_next = [[np.array([0,0]),np.array([0,0])]]
        self.Z_old = []
        self.Z_neigh = {f'{i}': [[np.eye(30)]] for i in self.neigh} #np.empty([20,20])

    
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
    def Tasks(self)->None:
        "Define the tasks separately"

        #n_robots = [self.degree+1, 0] # n° of neighbours + self agent
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
        
        self.task_input_limits = RobCont(omni=ca.vertcat(
              self.u.omni[0] - 4,   #vmax
            - self.u.omni[0] - 1,   #vmin
              self.u.omni[1] - 4,   #vmax
            - self.u.omni[1] - 1    #vmin
        ))
        
        self.task_input_min = RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1]))
        
        # ===========================Go-to-Goal====================================== #
        self.task_pos       = [None for i in range(len(self.goals))]
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
                (self.aux[0,0] - self.aux[1,0])**2 + (self.aux[0,1] - self.aux[1,1])**2 - 0,
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
        
        
        # =====================Collision Avoidance=================================== #
        threshold = 4
        self.aux_avoid_collision = ca.SX.sym('aux', 2, 2)
        self.mapping_avoid_collision = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
        self.task_avoid_collision = ca.vertcat(
            -(self.aux_avoid_collision[0,0] - self.aux_avoid_collision[1,0])**2 - (self.aux_avoid_collision[0,1] - self.aux_avoid_collision[1,1])**2,
        )
        self.task_avoid_collision_coeff = [
            #TaskBiCoeff(0, 0, 0, j, 1, -threshold**2) for j in self.robot_idx[1:]
            TaskBiCoeff(0, 0, 0, 1, 0, -threshold**2)
        ]
        if self.node_id == 1:
            self.mapping_avoid_collision = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
            self.task_avoid_collision_coeff = [
            #TaskBiCoeff(0, 0, 0, j, 1, -threshold**2) for j in self.robot_idx[1:]
            TaskBiCoeff(0, 0, 0, 1, 0, -threshold**2),
            #TaskBiCoeff(0, 0, 0, 2, 0, -threshold**2)
        ]
        

        # =====================Obstacle Avoidance===================================== #
        self.obstacle_pos = np.array([2,2])
        self.obstacle_size = 3
        # self.task_obs_avoidance = [ 
        #     ca.vertcat(- (self.s.omni[0] - self.obstacle_pos[0])**2 - (self.s.omni[0] - self.obstacle_pos[1])**2 + self.obstacle_size**2)
        # ]
        self.task_obs_avoidance = [ 
            ca.vertcat(- (self.s.omni[0] - self.obstacle_pos[0])**2 - (self.s.omni[1] - self.obstacle_pos[1])**2 + self.obstacle_size**2)
        ]

    def task_formation_method(self, agents, distance):
        aux = ca.SX.sym('aux', 2, 2)
        mapping = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
        task_formation = ca.vertcat(
            (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 0,
        )
        agents[0][0] = self.index_global_to_local(agents[0][0])  # convert global index to local index
        agents[0][1] = self.index_global_to_local(agents[0][1])  # convert global index to local index   
        task_formation_coeff = [
            TaskBiCoeff(0, agents[0][0], 0, agents[0][1], 0, distance**2),
        ]
        return aux, mapping, task_formation, task_formation_coeff

    # ---------------------------------------------------------------------------- #
    #                                      MPC                                     #
    # ---------------------------------------------------------------------------- #
    def MPC(self)->None:

        self.hompc = HOMPCMultiRobot(
            self.s.tolist(),
            self.u.tolist(),
            self.s_kp1.tolist(),
            self.n_robots.tolist(),
            self.degree
        )
        self.hompc.n_control = st.n_control
        self.hompc.n_pred = st.n_pred
        
        # ======================================================================== #

        self.create_tasks()
        '''for task in self.tasks:
            if task['name'] == "input_limits":
                self.hompc.create_task(
                    name = "input_limits", prio = task['prio'],
                    type = TaskType.Same,
                    ineq_task_ls = self.task_input_limits.tolist(),
                    robot_index= [self.robot_idx],
                    #ineq_task_coeff= self.task_input_limits_coeffs
                )
            elif task['name'] == "position":
                self.hompc.create_task(
                    name = "position", prio = task['prio'],
                    type = TaskType.Same,
                    eq_task_ls = self.task_pos[task['goal_index']].tolist(),
                    eq_task_coeff = self.task_pos_coeff[task['goal_index']].tolist(),
                    time_index = TaskIndexes.All,
                    robot_index= [[0]]
                )
            elif task['name'] == "input_minimization":
                self.hompc.create_task(
                    name = "input_minimization", prio = task['prio'],
                    eq_task_ls = self.task_input_min.tolist(),
                    robot_index= [self.robot_idx]
                )
            elif task['name'] == 'input_smooth':
                self.hompc.create_task(
                    name = "input_smooth", prio = task['prio'],
                    type = TaskType.SameTimeDiff,
                    ineq_task_ls = RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1])).tolist(),
                    #ineq_task_coeff = np.array([0,0,0,0]),
                    robot_index= [self.robot_idx]
                )
            elif task['name'] == 'formation':
                aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
                    task['agents'], task['distance']
                )
                self.hompc.create_task_bi(
                    name = "formation", prio = task['prio'],
                    type = TaskType.Bi,
                    aux = aux,
                    mapping = mapping.tolist(),
                    eq_task_ls = task_formation,
                    eq_task_coeff = task_formation_coeff,
                )
            elif task['name'] == 'collision_avoidance':
                self.hompc.create_task_bi(
                    name = "collision", prio = task['prio'],
                    type = TaskType.Bi,
                    aux = self.aux_avoid_collision,
                    mapping = self.mapping_avoid_collision.tolist(),
                    ineq_task_ls= self.task_avoid_collision,
                    ineq_task_coeff= self.task_avoid_collision_coeff,
                )
            elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                    name = "obstacle_avoidance", prio = task['prio'],
                    type = TaskType.Same,
                    ineq_task_ls = self.task_obs_avoidance,
                )
        
        for neigh in self.neigh_tasks:
            robot_idx = None
            for i in self.neigh:
                if neigh == f'agent_{i}':
                    robot_idx = self.robot_idx_global.index(i)
                    break
            if robot_idx is None:
                raise ValueError(f"Could not find robot index for neighbor {neigh}")
            for task in self.neigh_tasks[neigh]:
                if task['name'] == "position":
                    self.hompc.create_task(
                        name = "position", prio = task['prio'],
                        type = TaskType.Same,
                        eq_task_ls = self.task_pos[task['goal_index']].tolist(),
                        eq_task_coeff = self.task_pos_coeff[task['goal_index']].tolist(),
                        time_index = TaskIndexes.All,
                        robot_index= [[robot_idx]]
                    )   
                elif task['name'] == 'formation':
                    for t in task['agents']:
                        if is_formation_with_neigh(t, self.robot_idx_global):
                            aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
                                    task['agents'], task['distance']
                            )
                            self.hompc.create_task_bi(
                                name = "formation", prio = task['prio'],
                                type = TaskType.Bi,
                                aux = aux,
                                mapping = mapping.tolist(),
                                eq_task_ls = task_formation,
                                eq_task_coeff = task_formation_coeff,
                            )
                elif task['name'] == 'collision_avoidance':
                    self.hompc.create_task_bi(
                        name = "collision", prio = task['prio'],
                        type = TaskType.Bi,
                        aux = self.aux_avoid_collision,
                        mapping = self.mapping_avoid_collision.tolist(),
                        ineq_task_ls= self.task_avoid_collision,
                        ineq_task_coeff= self.task_avoid_collision_coeff,
                    )
                elif task['name'] == 'obstacle_avoidance':
                    self.hompc.create_task(
                        name = "obstacle_avoidance", prio = task['prio'],
                        type = TaskType.Same,
                        ineq_task_ls = self.task_obs_avoidance,
                    )'''
            
        # ======================================================================== #
        
        self.s = RobCont(omni=
            [np.array([1,1]) * (-2*self.node_id)
            for _ in range(self.n_robots.omni)],
            )
        if self.node_id == 1:
            self.s = RobCont(omni=
            [np.array([-1,-1]), np.array([0,0]) ],
            )

        self.s_history = [None for _ in range(self.n_steps)]
        
        self.s_init = copy.deepcopy(self.s)
        return
    
    # ---------------------------------------------------------------------------- #
    #                                Node's Methods                                #
    # ---------------------------------------------------------------------------- #

    def reorder_s_init(self, state_meas: list[float]):
        for j, s_j in enumerate(state_meas):
            if j in self.robot_idx_global:
                self.s_init.omni[self.index_global_to_local(j)] = copy.deepcopy(s_j) # TODO manage eterogeneous robots

        #update position of other robots (not neigh) seen as obstacles
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
        
    def receive_data(self, message)->None:
        " Append the received information in a local buffer"
        
        self.receiver.receive_message(message)
        
        #self.buffer.append(message)

    def transmit_data(self, receiver_id:int, update:str):
        " Create a message with primal variables state and the neighbours to share with"
        
        return self.sender.send_message(receiver_id, update)
        

    def update(self):          
        """Pop from local buffer the received dual variables of neighbours and minimize primal function"""
        
        if self.step != 0:
            self.rho_j = self.receiver.process_messages('D')
        
        if self.step < self.n_steps:
            
            print(self.step)
            rho_delta = self.rho_i - self.rho_j #! to be controlled
            
            self.u_star, self.y, cost = self.hompc(copy.deepcopy(self.s.tolist()), rho_delta)
            self.sender.y = copy.deepcopy(self.y)       # update copy of the states to share 
            
            self.cost_history.append(cost)
            # put in message u and s
            if self.step % self.a == 0:
                self.s = self.evolve(self.s_init, RobCont(omni=self.u_star[0]), self.dt)
                self.a = self.a * 1.5
            else:
                self.s = self.evolve(self.s, RobCont(omni=self.u_star[0]), self.dt)
            
            
            '''if self.step == 900 and (self.node_id==0):# or self.node_id==1):
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
                    )'''
                        
            print(f's:\t{self.s.tolist()}\n'
                  f'u:\t{self.u_star}\n')
               
            self.s_history[self.step] = copy.deepcopy(self.s.tolist())
            self.step += 1
                
        return 
        
    def dual_update(self):
        """Update the dual variables rho_i and rho_j using the received messages from neighbours"""
        
        if self.step > 0:
            self.save_data()
        
        self.y_j = self.receiver.process_messages('P')
                    
        # linear update of rho_i
        self.rho_i[0, :, :] += self.alpha * (np.tile(self.y_i[:, 0:self.n_xi], self.degree) - self.y_j[0, :, :])
        self.rho_i[1, :, :] += self.alpha * (self.y_i[:, self.n_xi:] - self.y_j[1, :, :])
        
        self.sender.rho = copy.deepcopy(self.rho_i)   # update copy of the states to share

        
    
    def evolve(self, s: list[list[float]], u_star: list[list[float]], dt: float):
        """Update the state of the system using the control input u_star and the time step dt"""

        n_intervals = 10
        
        for j, _ in enumerate(s.omni):
            
            for _ in range(n_intervals):
                s.omni[j] = s.omni[j] + dt / n_intervals * np.array([
                    u_star.omni[j][0],
                    u_star.omni[j][1],
                ])
        
        return s
    
    def plot_dual(self):
        return self.rho_i, self.neigh
    
    def save_data(self):
        if not st.save_data:
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
             
            
            writer.writerow(row)


    def create_tasks(self):
        """
            Create the tasks for the HOMPCMultiRobot instance
        """
        is_formation_with_neigh = lambda agents, neigh: all(item in neigh for item in agents)  # check if neighour's formation is with current agent's neighbour 


        for task in self.tasks:
            if task['name'] == "input_limits":
                self.hompc.create_task(
                    name = "input_limits", prio = task['prio'],
                    type = TaskType.Same,
                    ineq_task_ls = self.task_input_limits.tolist(),
                    robot_index= [self.robot_idx],
                    #ineq_task_coeff= self.task_input_limits_coeffs
                )
            elif task['name'] == "position":
                self.hompc.create_task(
                    name = "position", prio = task['prio'],
                    type = TaskType.Same,
                    eq_task_ls = self.task_pos[task['goal_index']].tolist(),
                    eq_task_coeff = self.task_pos_coeff[task['goal_index']].tolist(),
                    time_index = TaskIndexes.All,
                    robot_index= [[0]]
                )
            elif task['name'] == "input_minimization":
                self.hompc.create_task(
                    name = "input_minimization", prio = task['prio'],
                    eq_task_ls = self.task_input_min.tolist(),
                    robot_index= [self.robot_idx]
                )
            elif task['name'] == 'input_smooth':
                self.hompc.create_task(
                    name = "input_smooth", prio = task['prio'],
                    type = TaskType.SameTimeDiff,
                    ineq_task_ls = RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1])).tolist(),
                    #ineq_task_coeff = np.array([0,0,0,0]),
                    robot_index= [self.robot_idx]
                )
            elif task['name'] == 'formation':
                aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
                    task['agents'], task['distance']
                )
                self.hompc.create_task_bi(
                    name = "formation", prio = task['prio'],
                    type = TaskType.Bi,
                    aux = aux,
                    mapping = mapping.tolist(),
                    eq_task_ls = task_formation,
                    eq_task_coeff = task_formation_coeff,
                )
            elif task['name'] == 'collision_avoidance':
                self.hompc.create_task_bi(
                    name = "collision", prio = task['prio'],
                    type = TaskType.Bi,
                    aux = self.aux_avoid_collision,
                    mapping = self.mapping_avoid_collision.tolist(),
                    ineq_task_ls= self.task_avoid_collision,
                    ineq_task_coeff= self.task_avoid_collision_coeff,
                )
            elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                    name = "obstacle_avoidance", prio = task['prio'],
                    type = TaskType.Same,
                    ineq_task_ls = self.task_obs_avoidance,
                )
        
        for neigh in self.neigh_tasks:
            robot_idx = None
            for i in self.neigh:
                if neigh == f'agent_{i}':
                    robot_idx = self.robot_idx_global.index(i)
                    break
            if robot_idx is None:
                raise ValueError(f"Could not find robot index for neighbor {neigh}")
            for task in self.neigh_tasks[neigh]:
                if task['name'] == "position":
                    self.hompc.create_task(
                        name = "position", prio = task['prio'],
                        type = TaskType.Same,
                        eq_task_ls = self.task_pos[task['goal_index']].tolist(),
                        eq_task_coeff = self.task_pos_coeff[task['goal_index']].tolist(),
                        time_index = TaskIndexes.All,
                        robot_index= [[robot_idx]]
                    )   
                elif task['name'] == 'formation':
                    for t in task['agents']:
                        if is_formation_with_neigh(t, self.robot_idx_global):
                            aux, mapping, task_formation, task_formation_coeff = self.task_formation_method(
                                    task['agents'], task['distance']
                            )
                            self.hompc.create_task_bi(
                                name = "formation", prio = task['prio'],
                                type = TaskType.Bi,
                                aux = aux,
                                mapping = mapping.tolist(),
                                eq_task_ls = task_formation,
                                eq_task_coeff = task_formation_coeff,
                            )
                elif task['name'] == 'collision_avoidance':
                    self.hompc.create_task_bi(
                        name = "collision", prio = task['prio'],
                        type = TaskType.Bi,
                        aux = self.aux_avoid_collision,
                        mapping = self.mapping_avoid_collision.tolist(),
                        ineq_task_ls= self.task_avoid_collision,
                        ineq_task_coeff= self.task_avoid_collision_coeff,
                    )
                elif task['name'] == 'obstacle_avoidance':
                    self.hompc.create_task(
                        name = "obstacle_avoidance", prio = task['prio'],
                        type = TaskType.Same,
                        ineq_task_ls = self.task_obs_avoidance,
                    )


    def update_connection(self, adjacency_vector: np.array, neigh_tasks: dict):
        """
        Update the adjacency vector and the neighbour tasks.
        """
        
        self.adjacency_vector = adjacency_vector
        self.neigh = np.nonzero(adjacency_vector)[0].tolist()
        self.degree = len(self.neigh)
        self.n_robots = RobCont(omni=self.degree + 1)

        # expand the consensus variables
        self.y_i = np.pad(self.y_i, ((0, 0), (0, self.n_xi*(self.degree+1) - self.y_i.shape[1])), mode='constant', constant_values=0)
        self.rho_i = np.pad(self.rho_i, ((0, 0), (0,0), (0, self.n_xi*(self.degree) - self.rho_i.shape[2])), mode='constant', constant_values=0)
        self.rho_j = np.pad(self.rho_j, ((0, 0), (0,0), (0, self.n_xi*(self.degree) - self.rho_i.shape[2])), mode='constant', constant_values=0)
        self.y_j = np.pad(self.y_j, ((0, 0), (0,0), (0, self.n_xi*(self.degree) - self.rho_i.shape[2])), mode='constant', constant_values=0)
        #self.y_i = np.zeros((self.n_priority, self.n_xi*(self.degree+1)))
        #self.rho_i = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) 
        #np.random.rand(2, self.n_priority, self.n_xi*(self.degree))*0       # two values for rho_i and rho_j, n_properties rows, n_xi*(degree) columns
                                                                             # p1  [[[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p2  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p3  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...]]
        #self.y_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree)))   # p1  [[[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p2  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p3  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...]]
        #self.rho_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) # p1  [[[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
                                                                             # p2  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
                                                                             # p3  [[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...]]
        
        self.neigh_tasks.update(neigh_tasks) # expand dictionary with neighbour tasks