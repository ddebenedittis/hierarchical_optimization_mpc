import copy
import time

import numpy as np
import casadi as ca

from auxiliary.evolve import evolve
from ho_mpc.ho_mpc import HOMPC
from ho_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskIndexes, TaskType
from ho_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot 
from Message import *
from robot_models import get_unicycle_model, get_omnidirectional_model
import settings as st


class Node():
    """
    Representing the node and its actions and attributes

    Class method:
    -> transmit_data: create a message with information to share with neighbours
    -> update: pop from local memory buffer received messages and update its state
    -> receive_data: append the received message in a local memory buffer
    
    """

    def __init__(self, node_id: int, adjacency_vector: np.array, model: str, dt: float, tasks: list, goals: np.array, n_steps: int):

        super(Node, self).__init__()

        self.node_id = node_id  # ID of the node
        self.adjacency_vector = adjacency_vector # neighbours
        self.neigh = np.nonzero(adjacency_vector)[0].tolist() # index of neighbours
        self.degree = len(self.neigh) # numbers of neighbours
         
        self.x_init = np.random.randint(100) # initial state
        self.xi = self.x_init # state of the agent
        self.x_past = [] # evolution of the past states
        self.neighbors_sum = 0.0 # local neighbors_sum

        self.buffer = [] # local buffer to receive messages
        
        # ======================== Define The System Model ======================= #
    
        # Define the state and input variables, and the discrete-time dynamics model.
        
        self.dt = dt       # timestep size
        
        self.s = [None for _ in range(2)]
        self.u = [None for _ in range(2)]
        self.s_kp1 = [None for _ in range(2)]

        self.goals = goals

        self.s[0], self.u[0], self.s_kp1[0] = get_unicycle_model(self.dt)
        self.s[1], self.u[1], self.s_kp1[1] = get_omnidirectional_model(self.dt)

        #self.s, self.u, self.s_kp1 = model
        self.n_steps = n_steps
        self.step = 0
        self.tasks = tasks

        self.Xsym = [task['Xsym'] for task in self.tasks]
         
        self.s_next = [[np.array([0,0,0]),np.array([0,0,0])]]
        self.Z_old = []
        self.Z_neigh = {f'{i}': [[np.eye(20)]] for i in self.neigh} #np.empty([20,20])

    
    def receive_data(self, message)->None:
        " Append the received information in a local buffer"

        self.buffer.append(message)

    # ---------------------------------------------------------------------------- #
    #                                     Task                                     #
    # ---------------------------------------------------------------------------- #
    def Tasks(self)->None:
        # Define the tasks separately.

        n_robots = [self.degree+1, 0] # n° of neighbours + self agent

        tasks_creator = TasksCreatorHOMPCMultiRobot(
            self.s, self.u, self.s_kp1, self.dt, n_robots,
        )
        
        self.robot_idx = [[0],[]]

        self.task_input_limits = tasks_creator.get_task_input_limits()
        self.aux, self.mapping, self.task_formation, self.task_formation_coeff = tasks_creator.get_task_formation()

        self.task_pos       = [None for i in range(len(self.goals))]
        self.task_pos_coeff = [None for i in range(len(self.goals))]
        for i, g in enumerate(self.goals):
            self.task_pos[i], self.task_pos_coeff[i] = tasks_creator.get_task_pos_ref(
                [[g for n_j in range(n_robots[c])] for c in range(len(n_robots))], robot_idx=self.robot_idx
        )
        self.task_input_smooth, self.task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()
        self.task_input_limits_coeffs = [
            np.array([0, 0, 0, 0])
        ]
        
        self.task_vel_reference = ca.vertcat(
            (self.s_kp1[0] - self.s[0]) / self.dt - 1,
            (self.s_kp1[1] - self.s[1]) / self.dt - 0
        )
            
        self.task_input_min = tasks_creator.get_task_input_min()

    # ---------------------------------------------------------------------------- #
    #                                      MPC                                     #
    # ---------------------------------------------------------------------------- #
    def MPC(self)->None:
        self.hompc = HOMPCMultiRobot(self.s, self.u, self.s_kp1, [self.degree+1,0])
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

        # ======================================================================== #
        
        self.s = [[np.array([0,0,0]),np.array([0,0,0])]]
        
        
        
        self.s_history = [None] * self.n_steps

        return
    

    def transmit_data(self):
        " Create a message with state and the neighbours to share with"
        
        if self.step < 2:
            message = Message(self.node_id, self.xi, self.s_next, Z=None, Xsym=None)
        else :
            message = Message(self.node_id, self.xi, self.s_next, self.Z_old[-1], Xsym=None)
        

        return message, self.neigh

    
    def update(self):          
        """Pop from local buffer the receive states of neighbours and update its value"""
        
        self.x_past.append(self.xi)

        self.neighbors_sum = self.xi
        for j in self.neigh:
            data = self.buffer.pop()
            self.neighbors_sum += data.node_xi
            self.s[0][1] = (self.s[0][1] + data.s[0][0])/(self.degree + 1)
            if self.step >= 3 :
                self.null_sharing(data.Z, data.node_id)

            
         

        # update the local state
        self.xi = self.neighbors_sum/(self.degree + 1)

        if self.step < self.n_steps:
            print(self.step)
            
            self.u_star, self.s_next, Z= self.hompc(copy.deepcopy(self.s), self.Z_neigh)
            self.Z_old.append(Z)
                    
            self.s = evolve(self.s, self.u_star, self.dt)                      

            print(self.s)
            print(self.u_star)
            print(f'xi: {self.xi}')
            print()
            
            self.s_history[self.step] = copy.deepcopy(self.s)
            #self.s_history[self.step, :] = copy.deepcopy(self.s)
            self.step += 1
        
        if self.step > 2:
            self.hompc.null_consensus_start()
                
        return 
    
    def null_sharing(self, Z, i):
        self.Z_neigh[f'{i}'].append(Z)

    def s_update(self, s_neigh, i):
        # TODO 
        print('da fare')