import copy
import csv
import time
from time import sleep

import casadi as ca
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from matplotlib import pyplot as plt
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import dhqp_pkg.settings as st
from dhqp_pkg.disp_het_multi_rob import (
    MultiRobotArtistFlags,
    display_animation,
    save_snapshots,
)


def writer(filename, string):
    """
    inner function for logging
    """
    file = open(filename, 'a')  # "a" is for append
    file.write(string)
    file.close()


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__(
            'minimal_subscriber',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from launcher
        self.n_steps = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value
        self.n_nodes = self.get_parameter('N_AGENTS').value  # total number of agents
        self.dt = self.get_parameter('dt').value  # timestep size

        self.s_history = []
        self.u_history = []

        self.goals = st.goals
        self.step = 0
        self.step_plot = 0
        self.save_interval = 20
        self.t_0 = np.nan
        self.time = np.zeros(self.n_steps - 10) * np.nan
        self.start_time = time.time()
        # create logging file
        self.out_dir = self.get_parameter('out_dir').value
        self.filename = f'{self.out_dir}/traj_data.csv'
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            header = ['k']
            header.append('time')
            for i in range(st.n_nodes):
                header.append(f'stateX_{i}')
                header.append(f'stateY_{i}')
                header.append(f'stateRHO_{i}')
            for i in range(st.n_nodes):
                header.append(f'inputV{i}')
                header.append(f'inputOM{i}')

            writer.writerow(header)

        self.flags = MultiRobotArtistFlags()
        self.flags.voronoi = False

        # initialize subscription dict
        self.subscriptions_list = {}
        self.sub_input_list = {}
        # create a subscription to each neighbor
        for j in range(self.n_nodes):
            topic_name = f'/topic_{j}'
            self.subscriptions_list[j] = self.create_subscription(
                Float32MultiArray,
                topic_name,
                lambda msg, node=j: self.listener_callback(msg, node),
                20,  # Queue size for messages
            )
            topic_name = f'/robot_{j}/cmd_vel'
            self.subscriptions_list[j] = self.create_subscription(
                Twist,
                topic_name,
                self.input_callback,
                20,  # Queue size for messages
            )

        self.timer = self.create_timer(self.communication_time, self.timer_callback)

        # initialize a dictionary with the list of received messages from each neighbor j [a queue]
        self.received_data = {j: [] for j in range(self.n_nodes)}
        self.sync = False
        print(f'Setup of agent for graph plotting completed')

    def listener_callback(self, msg, node):
        self.received_data[node].append(list(msg.data))
        if all(self.received_data[j] for j in range(self.n_nodes)):
            self.sync = True

    def input_callback(self, msg):
        return

    def timer_callback(self):
        # Initialize a message of type float
        msg = Float32MultiArray()

        if self.sync:
            if np.isnan(self.t_0):
                self.t_0 = self.get_clock().now().nanoseconds

            # Reorder the vector of received messages from the agents
            self.time[self.step] = (self.get_clock().now().nanoseconds - self.t_0) / 1e9
            self.reorder_s_init(self.received_data)

            # self.get_logger().info(f'Iter:{self.step}\n s:{self.s_history[-1][0]}')
            # self.get_logger().info(f'u:{self.u_history[-1]}')
            self.get_logger().info(f'Iteration {self.step} under process')
            # update iteration counter
            self.sync = False

            if self.step % 20 == 0 and self.step > 0:
                with open(self.filename, mode='a', newline='') as file:
                    for s in range(self.step - self.save_interval, self.step):
                        if s == 0:
                            continue
                        writer = csv.writer(file)
                        flat_s = [v for sub in self.s_history[s][0] for v in sub]  # flatten
                        flat_u = [v for sub in self.u_history[s][0] for v in sub]  # flatten
                        row = [s] + [self.time[s]] + flat_s + flat_u
                        writer.writerow(row)
            self.step += 1
        # Stop the node if tt exceeds MAXITERS
        if self.step >= self.n_steps - 10:
            final_time = time.time() - self.start_time
            self.get_logger().info(f'Total simulation time: {final_time} seconds')
            # print('\nMAXITERS reached')
            # with open(self.filename, mode='a', newline='') as file:
            #     for s in range(1, self.step):
            #         writer = csv.writer(file)
            #         flat_s = [v for sub in self.s_history[s][0] for v in sub]  # flatten
            #         flat_u = [v for sub in self.u_history[s][0] for v in sub]  # flatten
            #         row = [s] + [self.time[s]] + flat_s + flat_u
            #         writer.writerow(row)
            if st.simulation:
                save_snapshots(
                    self.s_history,
                    None,
                    None,  # [[1.75, 0.28, 0.4]],  # [[3, 3, 0.5]],
                    st.dt,
                    [(self.step - 1) * st.dt],
                    f'{self.out_dir}/snapshot',
                    x_lim=[-1, 3.8],
                    y_lim=[-1, 2],
                    flags=self.flags,
                )
                display_animation(
                    self.s_history,
                    self.goals,
                    None,
                    st.dt,
                    st.visual_method,
                    video_name=f'{self.out_dir}/video.mp4',
                    x_lim=[-1.3, 3],
                    y_lim=[-1.3, 3],
                    flags=self.flags,
                )
            else:
                self.get_logger().info('My work is done, no plot requested. Goodbye!')

            self.destroy_node()

    def reorder_s_init(self, state_meas: list[float]):
        s = []
        u = []
        for j in range(self.n_nodes):
            s_j = [s for s in state_meas[j].pop(0)[1:]]
            u_j = s_j[-2:]
            s.append(s_j[:-2])
            u.append(u_j)
        self.s_history.append([s, []])
        self.u_history.append([u])


def main(args=None):
    rclpy.init(args=args)

    agent = MinimalSubscriber()
    print(f'Agent graph -- Waiting for sync.')
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
