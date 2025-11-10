import copy
import csv
import math
import subprocess
import threading
import time
from time import sleep

import casadi as ca
import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseArray, TwistStamped
from matplotlib import pyplot as plt
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion

import dhqp_pkg.settings as st
from dhqp_pkg.ho_mpc_multi_robot import (
    HOMPCMultiRobot,
    TaskBiCoeff,
    TaskIndexes,
    TaskType,
)
from dhqp_pkg.robot_models import (
    RobCont,
    get_omnidirectional_model,
    get_unicycle_model,
)


def writer(filename, string):
    """
    inner function for logging
    """
    file = open(filename, 'a')  # "a" is for append
    file.write(string)
    file.close()


class Agent(Node):
    def __init__(self):
        super().__init__(
            'agent',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from launcher
        self.node_id = self.get_parameter('agent_id').value
        self.adjacency_vector = np.array(self.get_parameter('neigh').value)
        self.neigh = np.nonzero(self.adjacency_vector)[0].tolist()
        self.degree = len(self.neigh)  # numbers of neighbours

        self.robot_idx_global = [self.node_id] + self.neigh
        self.robot_idx = [self.robot_idx_global.index(r) for r in self.robot_idx_global]
        print(f'global: {self.robot_idx_global}')
        print(f'local: {self.robot_idx}')

        self.n_steps = self.get_parameter('max_iters').value
        self.n_robots = RobCont(omni=self.degree + 1)  # number of robot seen from i
        self.n_nodes = self.get_parameter('N_AGENTS').value  # total number of agents

        self.dt = self.get_parameter('dt').value  # timestep size
        self.communication_time = 1e-4  # self.get_parameter('communication_time').value

        self.s = RobCont(omni=None, uni=None)  # symbolic state variables
        self.u = RobCont(omni=None, uni=None)
        self.s_kp1 = RobCont(omni=None, uni=None)

        # self.s.omni, self.u.omni, self.s_kp1.omni = get_omnidirectional_model(self.dt)
        self.s.omni, self.u.omni, self.s_kp1.omni = get_unicycle_model(self.dt)

        self.goals = st.goals
        self.step = 0
        self.step_plot = 0
        # self.tasks = self.get_parameter('system_tasks').value
        # self.neigh_tasks = self.get_parameter('neigh_tasks').value
        sys_tasks = st.system_tasks
        self.tasks = sys_tasks[f'agent_{self.node_id}']
        self.neigh_tasks = {}
        for j in self.neigh:
            self.neigh_tasks[f'agent_{j}'] = copy.deepcopy(sys_tasks[f'agent_{j}'])

        # create logging file
        self.out_dir = self.get_parameter('out_dir').value

        # ------------- PUBLISHER ------------#

        # create the publisher between node for communication
        self.publisher_ = self.create_publisher(
            Float32MultiArray,
            f'/topic_{self.node_id}',
            50,  # Queue size for messages
        )
        # create the publisher for optimal input computed
        self.ns = f'/robot_{self.node_id+1}'
        topic_name = f'{self.ns}/diff_drive_base_controller/cmd_vel'
        self.diff_drive_publisher = self.create_publisher(
            TwistStamped,
            topic_name,
            20,  # Queue size for messages
        )
        self.get_logger().info(f'Publisher created for {topic_name}')

        # Publisher for detected objects (as world positions)
        self.publisher = self.create_publisher(
            PoseArray,
            f'{self.ns}/detected_objects',
            20,
        )

        # ------------- SUBSCRIBER ------------#
        # initialize subscription dict
        self.subscriptions_list = {}
        # intra-neighbour communication subscribers
        for j in self.neigh:
            topic_name = f'/topic_{j}'
            self.subscriptions_list[j] = self.create_subscription(
                Float32MultiArray,
                topic_name,
                lambda msg, node=j: self.listener_callback(msg, node),
                20,  # Queue size for messages
            )
        # odometry subscriber
        self.subscription = self.create_subscription(
            ModelStates,
            '/model_states',  # topic name
            self.states_callback,
            20,
        )
        self.subscription = self.create_subscription(
            LaserScan,
            f'{self.ns}/scan',  # topic name
            self.scan_callback,
            20,
        )

        # Sensor configuration parameters
        self.gap_threshold = 5  # samples separating objects
        self.range_min = 0.3
        self.range_max = 8.0
        self.cylinder_radius = 0.5  # meters

        self.timer = self.create_timer(self.communication_time, self.timer_callback)

        # initialize a dictionary with the list of received messages from each neighbor j [a queue]
        self.received_data = {j: [] for j in self.robot_idx[1:]}

        # Create Tasks and MPC
        self.Tasks()
        self.MPC()

        print(f'Setup of agent {self.node_id} complete')

    def listener_callback(self, msg, node):
        self.received_data[self.index_global_to_local(node)].append(list(msg.data))

    def states_callback(self, msg):
        """Extract the pose from gazebo"""
        # Extract position from Gazebo
        for n, name in enumerate(msg.name):
            for neigh in self.robot_idx_global:
                if name == f'/robot_{neigh+1}':
                    x = msg.pose[n].position.x  # msg.pose.pose.position.x
                    y = msg.pose[n].position.y  # msg.pose.pose.position.y

                    # Extract orientation (quaternion -> yaw)
                    q = msg.pose[n].orientation  # msg.pose.pose.orientation
                    roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                    nn = self.index_global_to_local(neigh)
                    self.s.omni[nn] = np.array([x, y, yaw])  # self state

        # self.get_logger().info(f'Position: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad')

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        # Replace invalid or out-of-range values with NaN
        ranges[(ranges < self.range_min) | (ranges > self.range_max)] = np.nan

        # Identify valid indices (where the LiDAR sees something)
        valid_indices = np.where(~np.isnan(ranges))[0]

        if len(valid_indices) == 0:
            self.get_logger().info('No objects detected.')
            return

        # Group detections separated by >= gap_threshold samples
        object_groups = []
        current_group = [valid_indices[0]]

        for idx in valid_indices[1:]:
            if idx - current_group[-1] <= self.gap_threshold:
                current_group.append(idx)
            else:
                object_groups.append(current_group)
                current_group = [idx]
        object_groups.append(current_group)

        # Compute each object's average range and angle
        objects_local = []
        for group in object_groups:
            group_ranges = ranges[group]
            if np.all(np.isnan(group_ranges)):
                continue

            # Minimum range (closest point)
            min_idx_in_group = group[np.nanargmin(group_ranges)]
            min_range = ranges[min_idx_in_group]

            # Mean angle for this object (its approximate direction)
            group_angles = msg.angle_min + np.array(group) * msg.angle_increment
            mean_angle = np.mean(group_angles)

            objects_local.append((min_range, mean_angle))

        if not objects_local:
            self.get_logger().info('No valid object clusters found.')
            return

        # --- Convert each detected object's closest point to world coordinates ---
        x_r, y_r, yaw_r = self.s.omni[0]  # Robot's current position and orientation
        poses = PoseArray()
        poses.header = msg.header  # copy time and frame info
        poses.header.frame_id = 'my_world'

        for i, (r, mean_ang) in enumerate(objects_local):
            # Object center (1 m further along the beam)
            range_to_center = r + self.cylinder_radius
            # Position in robot frame
            x_local = range_to_center * math.cos(mean_ang)
            y_local = range_to_center * math.sin(mean_ang)
            # Transform to world frame
            x_world = x_r + x_local * math.cos(yaw_r) - y_local * math.sin(yaw_r)
            y_world = y_r + x_local * math.sin(yaw_r) + y_local * math.cos(yaw_r)

            self.objects_global = [x_world, y_world]
            # Save as Pose (only position is relevant)
            pose = Pose()
            pose.position.x = x_world
            pose.position.y = y_world
            pose.position.z = 0.0
            poses.poses.append(pose)

        # Publish all detected objects
        self.publisher.publish(poses)
        # if self.step < 4:
        #     self.obstacle_pos = self.objects_global
        #     self.task_obs_avoidance = [
        #         ca.vertcat(
        #             -((self.s.omni[0][0] - self.obstacle_pos[0]) ** 2)
        #             - (self.s.omni[0][1] - self.obstacle_pos[1]) ** 2
        #             + self.obstacle_size**2
        #         )
        #     ]
        #     self.hompc.update_task_bi(
        #             name='obstacle_avoidance',
        #             prio=3,
        #             type=TaskType.Bi,
        #             ineq_task_coeff=self.task_obs_avoidance,
        #     )
        # --- Print all objects’ closest points ---
        """self.get_logger().info(f"Detected {len(objects)} objects:")
        for i, (dist, ang) in enumerate(objects):
            self.get_logger().info(
                f"  Object {i+1}: distance = {dist:.2f} m, angle = {np.degrees(ang):.1f}°"
            )"""

        # Find the closest object
        """closest = min(objects, key=lambda x: x[0])
        distance, angle = closest

        self.get_logger().info(
            f"Closest object: distance = {distance:.2f} m, angle = {np.degrees(angle):.1f}°"
        )"""

    def timer_callback(self):
        # Perform Partitioned optimization
        # Initialize a message of type float
        msg = Float32MultiArray()
        command = TwistStamped()

        if self.step == 0:  # Let the publisher start at the first iteration
            msg.data = [float(self.step)]
            [msg.data.append(float(ss)) for ss in self.s.omni[0]]
            self.publisher_.publish(msg)

            # log files
            # 1) visualize on the terminal
            self.get_logger().info(f'Iter:{self.step} s:{self.s.tolist()}')
            self.step += 1

        else:  # Have all messages at time t-1 arrived?
            # Check if lists are nonempty
            all_received = all(
                self.received_data[j] for j in self.robot_idx[1:]
            )  # check if all neighbors' have been received
            sync = False

            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(
                    self.step - 1 == self.received_data[j][0][0] for j in self.robot_idx[1:]
                )  # True if all True

            if sync:
                # Reorder the state vector received from the neighbors
                self.reorder_s_init(self.received_data)
                # if self.step == 1:
                #     self.s.omni[1:] = copy.deepcopy(self.s_init.omni[1:])

                self.u_star, self.y, self.cost_p = self.hompc(copy.deepcopy(self.s.tolist()))

                # self.s = self.evolve(copy.deepcopy(self.s), RobCont(omni=self.u_star[0]), self.dt)

                # publish the command
                command.header.stamp = self.get_clock().now().to_msg()
                command.twist.linear.x = float(self.u_star[0][0][0])
                command.twist.angular.z = float(self.u_star[0][0][1])
                self.diff_drive_publisher.publish(command)

                # publish the updated message
                msg.data = [float(self.step)]
                [msg.data.append(float(ss)) for ss in self.s.omni[0]]
                self.publisher_.publish(msg)
                self.get_logger().info(
                    f'Iter:{self.step}\n s:{self.s.tolist( )} u:{self.u_star[0]}\n'
                )

                # Stop the node if tt exceeds MAXITERS
                if self.step > self.n_steps:
                    print('\nMAXITERS reached')
                    command.header.stamp = self.get_clock().now().to_msg()
                    command.twist.linear.x = float(0)
                    command.twist.angular.z = float(0)
                    self.diff_drive_publisher.publish(command)
                    self.destroy_node()

                # update iteration counter
                self.step += 1

    # ---------------------------------------------------------------------------- #
    #                                     Task                                     #
    # ---------------------------------------------------------------------------- #

    def Tasks(self) -> None:
        "Define the tasks separately"

        # =========================== Define The Tasks ========================== #

        self.task_input_limits = RobCont(
            omni=ca.vertcat(
                self.u.omni[0] - st.v_max,  # vmax
                -self.u.omni[0] + st.v_min,  # vmin
                self.u.omni[1] - st.omega_max,  # vmax
                -self.u.omni[1] + st.omega_min,  # vmin
            )
        )

        self.task_input_min = RobCont(omni=ca.vertcat(self.u.omni[0], self.u.omni[1]))

        # ===========================Go-to-Goal====================================== #
        self.task_pos = [None for i in range(len(self.goals))]
        self.task_pos_coeff = [None for i in range(len(self.goals))]
        for i, g in enumerate(self.goals):
            self.task_pos[i] = RobCont(omni=ca.vertcat(self.s_kp1.omni[0], self.s_kp1.omni[1]))
            self.task_pos_coeff[i] = RobCont(
                omni=[[np.array(g)] for _ in range(self.n_robots.omni)],
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
        self.threshold = 0.8
        self.aux_avoid_collision = ca.SX.sym('aux', 2, 2)
        self.mapping_avoid_collision = RobCont(omni=ca.vertcat(self.s.omni[0], self.s.omni[1]))
        self.task_avoid_collision = ca.vertcat(
            -((self.aux_avoid_collision[0, 0] - self.aux_avoid_collision[1, 0]) ** 2)
            - (self.aux_avoid_collision[0, 1] - self.aux_avoid_collision[1, 1]) ** 2,
        )

        self.task_avoid_collision_coeff = [
            TaskBiCoeff(0, i, 0, j, 0, -(self.threshold**2))
            for i in range(self.n_robots.omni)
            for j in range(i + 1, self.n_robots.omni)
        ]

        # =====================Obstacle Avoidance===================================== #

        self.obstacle_pos = np.array([0, 0])
        self.obstacle_size = 2
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
                    # time_index=TaskIndexes.All,
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
            elif task['name'] == 'obstacle_avoidance':
                self.hompc.create_task(
                    name='obstacle_avoidance',
                    prio=task['prio'],
                    type=TaskType.Same,
                    ineq_task_ls=self.task_obs_avoidance,
                )

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
                    # time_index=TaskIndexes.All,
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
            self.s = RobCont(omni=[np.array([-5, -5, 0]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 1:
            self.s = RobCont(omni=[np.array([5, -5, -3]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 2:
            self.s = RobCont(omni=[np.array([5, 5, -3]) for _ in range(self.n_robots.omni)])
        elif self.node_id == 3:
            self.s = RobCont(omni=[np.array([-5, 5, 0]) for _ in range(self.n_robots.omni)])

        self.s_history = [None for _ in range(self.n_steps)]
        self.s_history_p = [None for _ in range(self.n_steps)]
        self.s_init = copy.deepcopy(self.s)
        return

    # ---------------------------------------------------------------------------- #
    #                                     Methods                                  #
    # ---------------------------------------------------------------------------- #
    def reorder_s_init(self, state_meas: list[float]):
        # self.s_init.omni[0] = copy.deepcopy(self.s.omni[0])  # self state

        for j in self.robot_idx[1:]:
            s_j = [s for s in state_meas[j].pop(0)[1:]]
            # s_j = np.array(s_j)
            # self.s.omni[j] = copy.deepcopy(s_j)

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
        #         s.omni[j] = s.omni[j] + dt / n_intervals * np.array(
        #             [
        #                 u_star.omni[j][0],
        #                 u_star.omni[j][1],
        #             ]
        #         )

        return s

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


def main(args=None):
    rclpy.init(args=args)

    agent = Agent()
    print(f'Agent {agent.node_id} -- Waiting for sync.')
    sleep(0.5)
    print('GO!')
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print('----- Node stopped cleanly -----')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
