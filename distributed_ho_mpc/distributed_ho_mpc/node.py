import copy
import time

import numpy as np
import casadi as ca

from hierarchical_optimization_mpc.auxiliary.evolve import evolve
from hierarchical_optimization_mpc.utils.robot_models import get_unicycle_model, get_omnidirectional_model, RobCont

from ho_mpc.ho_mpc import HOMPC
from ho_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskIndexes, TaskType
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
         
        self.buffer = [] # local buffer to receive primal variables
        self.buffer_dual = [] # local buffer to receive dual variables
        self.x_neigh = [] # local buffer to store primal variables to share
        self.x_i = [] 
        self.n_priority = 3 # number of priorities
        self.n_xi = 2 # dimension of primal variables

        # ======================== Variables updater ======================= #
        self.alpha = 1e-6 * np.ones(self.n_xi * (self.degree)) # step size for primal and dual variables
        
        
        self.y_i = np.zeros((self.n_priority, self.n_xi*(self.degree+1)))
        self.rho_i = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) # two values for rho_i and rho_j, n_properties rows, n_xi*(degree) columns
                                                                             # p1 [[[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p2  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...],
                                                                             # p3  [[rho^(ij1)_i, rho^(ij1)_j1], [rho^(ij2)_i, rho^(ij2)_j2]...]]
        self.y_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree+1))) # p1 [[[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p2  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...],
                                                                             # p3  [[x^(j1)_i, x^(j1)_j], [x^(j2)_i, x^(j2)_j]...]]
        self.rho_j = np.zeros((2, self.n_priority, self.n_xi*(self.degree))) # p1 [[[rho^(j1i)_i, rho^(j1i)_j1], [rho^(j2i)_i, rho^(j2i)_j2]...],
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
    
        self.s.omni, self.u.omni, self.s_kp1.omni = get_omnidirectional_model(dt)
        
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

        self.u_next = [[np.array([0,0]),np.array([0,0])]]
        self.Z_old = []
        self.Z_neigh = {f'{i}': [[np.eye(30)]] for i in self.neigh} #np.empty([20,20])

    
    # ---------------------------------------------------------------------------- #
    #                                     Task                                     #
    # ---------------------------------------------------------------------------- #
    def Tasks(self)->None:
        "Define the tasks separately"

        #n_robots = [self.degree+1, 0] # n° of neighbours + self agent

        self.tasks_creator = TasksCreatorHOMPCMultiRobot(
            self.s.tolist(),
            self.u.tolist(),
            self.s_kp1.tolist(),
            self.n_robots.tolist(),
        )
        
        self.robot_idx = [self.node_id] + self.neigh

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
        #                                     )


    # ---------------------------------------------------------------------------- #
    #                                      MPC                                     #
    # ---------------------------------------------------------------------------- #
    def MPC(self)->None:
        self.hompc = HOMPCMultiRobot(
            self.s.tolist(),
            self.u.tolist(),
            self.s_kp1.tolist(),
            self.n_robots.tolist(),
        )
        self.hompc.n_control = st.n_control
        self.hompc.n_pred = st.n_pred
        
        for task in self.tasks:
            if task['name'] == "input_limits":
                self.hompc.create_task(
                                name = "input_limits", prio = task['prio'],
                                type = TaskType.Same,
                                ineq_task_ls = self.task_input_limits,
                                #ineq_task_coeff= self.task_input_limits_coeffs
                                )
            elif task['name'] == "position":
                self.hompc.create_task(
                                name = "position", prio = task['prio'],
                                type = TaskType.Same,
                                eq_task_ls = self.task_pos[task['goal_index']],
                                eq_task_coeff = self.task_pos_coeff[task['goal_index']],
                                time_index = TaskIndexes.All,
                                robot_index=self.robot_idx
                                )
            elif task['name'] == "input_minimization":
                self.hompc.create_task(
                                name = "input_minimization", prio = task['prio'],
                                eq_task_ls = self.task_input_min,
                                )          
            elif task['name'] == "reference":
                self.hompc.create_task(
                                name = "reference", prio = task['prio'],
                                eq_task_ls = self.task_vel_reference,
                                )
            elif task['name'] == 'input_smooth':
                self.hompc.create_task(
                                name = "input_smooth", prio = task['prio'],
                                type = TaskType.SameTimeDiff,
                                ineq_task_ls = self.task_input_smooth,
                                ineq_task_coeff = self.task_input_smooth_coeffs
                                        )
            elif task['name'] == 'formation':
                self.hompc.create_task_bi(
                                name = "formation", prio = task['prio'],
                                type = TaskType.Bi,
                                aux = self.aux,
                                mapping = self.mapping,
                                eq_task_ls = self.task_formation,
                                eq_task_coeff = self.task_formation_coeff,
                                        )
            '''elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                                name = "obstacle_avoidance", prio = task['prio'],
                                type = TaskType.Same,
                                eq_task_ls = self.task_obs_avoidance,
                                robot_index=self.robot_idx,
                                time_index = TaskIndexes.All,
                                        )'''
        for neigh in self.neigh_tasks:
            for i in self.neigh:
                if neigh == f'agent_{i}':
                    robot_idx = i
            for task in self.neigh_tasks[neigh]:
                if task['name'] == "input_limits":
                    self.hompc.create_task(
                                    name = "input_limits", prio = task['prio'],
                                    type = TaskType.Same,
                                    ineq_task_ls = self.task_input_limits,
                                    #robot_index=robot_idx,
                                    #ineq_task_coeff= self.task_input_limits_coeffs
                                    )
                elif task['name'] == "position":
                    self.hompc.create_task(
                                    name = "position", prio = task['prio'],
                                    type = TaskType.Same,
                                    eq_task_ls = self.task_pos[task['goal_index']],
                                    eq_task_coeff = self.task_pos_coeff[task['goal_index']],
                                    time_index = TaskIndexes.All,
                                    robot_index=robot_idx
                                    )
                elif task['name'] == "input_minimization":
                    self.hompc.create_task(
                                    name = "input_minimization", prio = task['prio'],
                                    eq_task_ls = self.task_input_min,
                                    robot_index=robot_idx
                                    )          
                elif task['name'] == "reference":
                    self.hompc.create_task(
                                    name = "reference", prio = task['prio'],
                                    eq_task_ls = self.task_vel_reference,
                                    robot_index=robot_idx
                                    )
                elif task['name'] == 'input_smooth':
                    self.hompc.create_task(
                                    name = "input_smooth", prio = task['prio'],
                                    type = TaskType.SameTimeDiff,
                                    ineq_task_ls = self.task_input_smooth,
                                    ineq_task_coeff = self.task_input_smooth_coeffs,
                                    robot_index=robot_idx        
                                            )
                elif task['name'] == 'formation':
                    self.hompc.create_task_bi(
                                    name = "formation", prio = task['prio'],
                                    type = TaskType.Bi,
                                    aux = self.aux,
                                    mapping = self.mapping,
                                    eq_task_ls = self.task_formation,
                                    eq_task_coeff = self.task_formation_coeff,
                                            )
                '''elif task['name'] == 'obstacle_avoidance':
                    self.hompc.create_task(
                                    name = "obstacle_avoidance", prio = task['prio'],
                                    type = TaskType.Same,
                                    eq_task_ls = self.task_obs_avoidance,
                                    robot_index=robot_idx,
                                    time_index = TaskIndexes.All,
                                            )'''
            

        # ======================================================================== #
        
        self.s = RobCont(omni=
        [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
         for _ in range(self.n_robots.omni)],
        )
        #self.s = [[np.array([0,0,0]),np.array([0,0,0])]]
        
        self.s_history = [None for _ in range(self.n_steps)]

        return
    
    # ---------------------------------------------------------------------------- #
    #                                Node's Methods                                #
    # ---------------------------------------------------------------------------- #


    def receive_data(self, message)->None:
        " Append the received information in a local buffer"
        
        self.receiver.receive_message(message)
        
        #self.buffer.append(message)

    def transmit_data(self, receiver_id:int, update:str):
        " Create a message with primal variables state and the neighbours to share with"
        
        return self.sender.send_message(receiver_id, update)
        
        # modify message 
        # message = []
        # for jj in self.x_neigh:
        #     message.append(Message(self.node_id, self.x_i, jj))
        # return message, self.neigh

    def update(self):          
        """Pop from local buffer the received dual variables of neighbours and minimize primal function"""
        
        """for j in self.neigh:
            data = self.buffer_dual.pop()
            if self.step >= 3 :
                #self.null_sharing(data.Z, data.node_id)
                
                # consensus on x[s,u] 
                # for c, n_r in enumerate(self.n_robots):
                #     for j in range(self.n_robots[c]):
                #         for k in range(st.n_control):
                #             self.s_opt[c][j][k] = copy.deepcopy((self.s_opt[c][j][k] + data.s[c][np.abs(j-1)][k])/(self.degree+1))  #incrocio i due vettori di ottimizzazione
                # for c, n_r in enumerate(self.n_robots):
                #     for j in range(self.n_robots[c]):
                #         for k in range(st.n_control):
                #             self.u_opt[c][j][k] = copy.deepcopy((self.u_opt[c][j][k] + data.u[c][np.abs(j-1)][k])/(self.degree+1))
                #self.s[0] = [self.s_opt[0][0][0],self.s_opt[0][1][0]] # update new value of s"""
        self.rho_j = self.receiver.process_messages('D')
        
        if self.step < self.n_steps:
            print(self.step)
            rho_delta = self.rho_i - self.rho_j #! to be controlled
            #self.u_star, self.u_opt, self.s_opt, Z= self.hompc(copy.deepcopy(self.s.tolist()), self.Z_neigh, copy.deepcopy(self.u_opt.tolist()), self.node_id)

            self.u_star, self.y = self.hompc(copy.deepcopy(self.s.tolist()), rho_delta, self.Z_neigh)
            self.sender.y = copy.deepcopy(self.y)       # update copy of the states to share 
            
            # put in message u and s
            self.s = evolve(self.s, RobCont(omni=self.u_star[0]), self.dt)
            
            print(f's:\t{self.s}\n'
                  f'u:\t{self.u_star}\n')
            
               
            self.s_history[self.step] = copy.deepcopy(self.s)
            #self.s_history[self.step, :] = copy.deepcopy(self.s)
            self.step += 1
        
        if self.step > 2:
            self.hompc.null_consensus_start()
                
        return 
        
    def dual_update(self):
        # TODO
        self.y_j = self.receiver.process_messages('P')
        
        # linear update of rho_i
        self.rho_i[0, :, :] += self.alpha * (np.tile(self.y_i[:, 0:self.n_xi], self.degree) - self.y_j[0, :, :])
        self.rho_i[1, :, :] += self.alpha * (self.y_i[:, self.n_xi+1:-1] - self.y_j[1, :, self.n_xi+1:-1])
        
        self.sender.rho = copy.deepcopy(self.rho_i)   # update copy of the states to share
        
        
    def null_sharing(self, Z, i):
        self.Z_neigh[f'{i}'].append(Z)

    
