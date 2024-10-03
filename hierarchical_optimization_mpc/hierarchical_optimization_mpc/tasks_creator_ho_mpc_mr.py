import casadi as ca
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import TaskBiCoeff
from hierarchical_optimization_mpc.voronoi_task import VoronoiTask



class TasksCreatorHOMPCMultiRobot():
    """
    Helper class to create tasks for the HOMPCMultiRobot class when using
    unicycles and omnidirectional robots.
    """
    
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
        
        self.bounding_box = None
        
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
                  self.u[1][1] - self.v_max,
                - self.u[1][1] + self.v_min
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
            [[np.array([0.95, 0.95, 0.9, 0.9])] for j in range(self.n_robots[0])],
            [[np.array([0.95, 0.95, 0.9, 0.9])] for j in range(self.n_robots[1])],
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

    def get_task_pos_ref(self, pos_ref, robot_idx: list[list[int]] = None):
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
        
        if robot_idx is None:
            task_pos_ref_coeff = [
                [
                    [(pos_ref[c][j][0:2]).flatten()] for j in range(self.n_robots[c])
                ] for c in range(len(self.n_robots))
            ]
        else:
            task_pos_ref_coeff = [
                [
                    [(pos_ref[c][j][0:2]).flatten()] for j in robot_idx[c]
                ] for c in range(len(robot_idx))
            ]
        
        return task_pos_ref, task_pos_ref_coeff
    
    # =========================== Get_task_coverage ========================== #
    
    def get_task_coverage(self, robot_idx: list[list[int]] = None):        
        if robot_idx is None:
            towers = np.array(
                [e[0:2] for e in self.states_bar[0]] +
                [e[0:2] for e in self.states_bar[1]]
            )
            n_cov = self.n_robots
        else:
            towers = np.array(
                [self.states_bar[0][j][0:2] for j in robot_idx[0]] +
                [self.states_bar[0][j][0:2] for j in robot_idx[1]]
            )
            n_cov = [len(robot_idx[0]), len(robot_idx[1])]

        vor_task = VoronoiTask(towers, self.bounding_box)
        
        pos_ref = [[np.array([0, 0]) for _ in range(n_cov[c])] for c in range(len(n_cov))]
        for c in range(len(n_cov)):
            for i in range(len(pos_ref[c])):
                pos_ref[c][i] = vor_task.centroids[i + sum(n_cov[0:c]), :]
            
        task_cov, task_cov_coeff = self.get_task_pos_ref(
            pos_ref, robot_idx
        )
                
        return task_cov, task_cov_coeff
    
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
        
    # ======================== Get_task_obs_avoidance ======================== #
        
    def get_task_obs_avoidance(self, obstacle_pos: np.ndarray, threshold: float = 1.):
        if len(obstacle_pos) != 2:
            raise ValueError("The obstacle position must be a vector of size 2.")
        
        return [
            ca.vertcat(- (self.s[0][0] - obstacle_pos[0])**2 - (self.s[0][1] - obstacle_pos[1])**2 + threshold**2),
            ca.vertcat(- (self.s[1][0] - obstacle_pos[0])**2 - (self.s[1][1] - obstacle_pos[1])**2 + threshold**2),
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
            TaskBiCoeff(1, 0, 1, 1, 3, 3**2),
            TaskBiCoeff(1, 1, 1, 2, 3, 3**2),
            TaskBiCoeff(1, 2, 1, 3, 3, 3**2),
            TaskBiCoeff(1, 3, 1, 0, 3, 3**2),
            TaskBiCoeff(1, 1, 1, 3, 3, 2*3**2),
            TaskBiCoeff(1, 0, 1, 2, 3, 2*3**2),
            # TaskBiCoeff(0, 1, 1, 0, 3, 2*3**2),
            TaskBiCoeff(1, 4, 1, 0, 3, 3**2/2),
            TaskBiCoeff(1, 4, 1, 1, 3, 3**2/2),
            TaskBiCoeff(1, 4, 1, 2, 3, 3**2/2),
            TaskBiCoeff(1, 4, 1, 3, 3, 3**2/2),
        ]
        
        return aux, mapping, task_formation, task_formation_coeff
    