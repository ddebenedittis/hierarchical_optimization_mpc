import casadi as ca
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import TaskBiCoeff



class TasksCreatorHOMPCMultiRobot():
    def __init__(
        self, states: list[ca.SX], inputs: list[ca.SX],
        fs: list[ca.SX], dt: float,
        n_robots: list[int],
        states_bar = None, inputs_bar = None
    ) -> None:
        self.s = states
        self.u = inputs 
        self.s_kp1 = fs
        self.dt = dt
        self.n_robots = n_robots
        
        if states_bar is None:
            self.states_bar = [
                [np.zeros(states[c].size1()) for n_j in range(n_robots[c])]
            for c in range(len(n_robots))]
        else:
            self.states_bar = states_bar
        
        if inputs_bar is None:
            self.inputs_bar = [
                [np.zeros(inputs[c].size1()) for n_j in range(n_robots[c])]
            for c in range(len(n_robots))]
        else:
            self.inputs_bar = inputs_bar
        
        # Input limits
        self.v_max = 5
        self.v_min = -5
        self.omega_max = 1
        self.omega_min = -1
        
    # ========================= Get_task_input_limits ======================== #
        
    def get_task_input_limits(self):
        return [
            ca.vertcat(
                  self.u[0][0] - self.v_max,
                - self.u[0][0] + self.v_min,
                  self.u[0][1] - self.omega_max,
                - self.u[0][1] + self.omega_min
            ),
            ca.vertcat(
                  self.u[1][0] - self.v_max,
                - self.u[1][0] + self.v_min,
                  self.u[1][1] - self.omega_max,
                - self.u[1][1] + self.omega_min
            ),
        ]
        
    # ========================= Get_task_input_smooth ======================== #
    
    def get_task_input_smooth(self):
        task_input_smooth = [
            ca.vertcat(
                  self.u[0][0],
                - self.u[0][0],
                  self.u[0][1],
                - self.u[0][1]
            ),
            ca.vertcat(
                  self.u[1][0],
                - self.u[1][0],
                  self.u[1][1],
                - self.u[1][1]
            ),
        ]
        
        task_input_smooth_coeffs = [
            [[np.array([0.9, 0.9, 0.8, 0.8])] for j in range(self.n_robots[0])],
            [[np.array([0.9, 0.9, 0.8, 0.8])] for j in range(self.n_robots[1])],
        ]
        
        return task_input_smooth, task_input_smooth_coeffs
        
    # ======================= Get_task_centroid_vel_ref ====================== #
        
    def get_task_centroid_vel_ref(self, ref):
        assert len(ref) == 2
        
        return [
            ca.vertcat(
                (self.s_kp1[0][0] - self.s[0][0]) / self.dt - ref[0],
                (self.s_kp1[0][1] - self.s[0][1]) / self.dt - ref[1],
            ) / sum(self.n_robots),
            ca.vertcat(
                (self.s_kp1[1][0] - self.s[1][0]) / self.dt - ref[0],
                (self.s_kp1[1][1] - self.s[1][1]) / self.dt - ref[1],
            ) / sum(self.n_robots),
        ]
        
    # =========================== Get_task_vel_ref =========================== #
        
    def get_task_vel_ref(self, ref):
        task_vel_ref = [
            ca.vertcat(
                (self.s_kp1[0][0] - self.s[0][0]) / self.dt - ref[0],
                (self.s_kp1[0][1] - self.s[0][1]) / self.dt - ref[1],
            ),
            ca.vertcat(
                (self.s_kp1[1][0] - self.s[1][0]) / self.dt - ref[0],
                (self.s_kp1[1][1] - self.s[1][1]) / self.dt - ref[1],
            ),
        ]
        
        task_vel_ref_coeff = [
            [
                [np.array([ 2.0,  0])],
                [np.array([ 2.0,  2.0])],
                [np.array([ 0,    2.0])],
            ],
            [[]],
        ]
        
        return task_vel_ref, task_vel_ref_coeff
    
    # =========================== Get_task_pos_ref =========================== #

    def get_task_pos_ref(self, pos_ref):
        task_pos_ref = [
            ca.vertcat(
                self.s_kp1[0][0],
                self.s_kp1[0][1],
            ),
            ca.vertcat(
                self.s_kp1[1][0],
                self.s_kp1[1][1],
            ),
        ]
        
        task_pos_ref_coeff = [
            [
                [(pos_ref[c][j][0:2]).flatten()] for j in range(self.n_robots[c])
            ] for c in range(len(self.n_robots))
        ]
        
        return task_pos_ref, task_pos_ref_coeff
    
    # ========================== Get_task_input_min ========================== #
    
    def get_task_input_min(self, ):
        return [
            ca.vertcat(
                self.u[0][0],
                self.u[0][1],
            ),
            ca.vertcat(
                  self.u[1][0],
                - self.u[1][0],
                  self.u[1][1],
                - self.u[1][1]
            ),
        ]
        
    # ======================= Get_task_avoid_collision ======================= #
    
    def get_task_avoid_collision(self, threshold: float = 0.1):
        aux = ca.SX.sym('aux', 2, 2)
    
        mapping = [
            ca.vertcat(
                self.s[0][0],
                self.s[0][1],
            ),
            ca.vertcat(
                self.s[1][0],
                self.s[1][1],
            ),
        ]
        
        task_avoid_collision = ca.vertcat(
            (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - threshold**2,
        )
        
        task_avoid_collision_coeff = [
            TaskBiCoeff(0, 0, 0, 1, 1, 0),
            TaskBiCoeff(0, 0, 0, 2, 1, 0),
            TaskBiCoeff(0, 1, 0, 2, 1, 0),
        ]
        
        return aux, mapping, task_avoid_collision, task_avoid_collision_coeff
    
    # ========================== Get_task_formation ========================== #
    
    def get_task_formation(self, ref = None):
        aux = ca.SX.sym('aux', 2, 2)
    
        mapping = [
            ca.vertcat(
                self.s[0][0],
                self.s[0][1],
            ),
            ca.vertcat(
                self.s[1][0],
                self.s[1][1],
            ),
        ]
        
        # task_avoid_collision = ca.vertcat(
        #     (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 5,
        # )
        
        task_formation = ca.vertcat(
            (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 0,
        )
        
        task_formation_coeff = [
            TaskBiCoeff(0, 0, 0, 1, 1, 10),
            TaskBiCoeff(0, 0, 0, 2, 1, 10),
            TaskBiCoeff(0, 1, 0, 2, 1, 10),
        ]
        
        return aux, mapping, task_formation, task_formation_coeff
    