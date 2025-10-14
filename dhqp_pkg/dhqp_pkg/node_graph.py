import copy
import csv
from time import sleep

import casadi as ca
import numpy as np
import rclpy
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


class Agent(Node):
    def __init__(self):
        super().__init__(
            'agent',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Get parameters from launcher
        self.n_steps = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value * 4
        self.n_nodes = self.get_parameter('N_AGENTS').value  # total number of agents
        self.dt = self.get_parameter('dt').value  # timestep size

        self.s_history = []

        self.goals = st.goals
        self.step = 0
        self.step_plot = 0

        # create logging file
        self.out_dir = self.get_parameter('out_dir').value
        # self.filename = f'{self.out_dir}/node_{self.node_id}_data.csv'
        # self.filename = "_csv_file/agent_{}.csv".format(self.node_id)
        # file = open(self.filename, "w+")  # 'w+' needs to create file and open in writing mode if doesn't exist
        # file.close()

        self.flags = MultiRobotArtistFlags()
        self.flags.voronoi = False

        # initialize subscription dict
        self.subscriptions_list = {}

        # create a subscription to each neighbor
        for j in range(self.n_nodes):
            topic_name = f'/topic_{j}'
            self.subscriptions_list[j] = self.create_subscription(
                Float32MultiArray,
                topic_name,
                lambda msg, node=j: self.listener_callback(msg, node),
                20,  # Queue size for messages
            )

        self.timer = self.create_timer(self.communication_time, self.timer_callback)

        # initialize a dictionary with the list of received messages from each neighbor j [a queue]
        self.received_data = {j: [] for j in range(self.n_nodes)}

        print(f'Setup of agent for graph plotting completed')

    def listener_callback(self, msg, node):
        self.received_data[node].append(list(msg.data))

    def timer_callback(self):
        # Initialize a message of type float
        msg = Float32MultiArray()

        if self.step == 0:  # Let the publisher start at the first iteration
            # log files
            # 1) visualize on the terminal
            all_received = all(
                self.received_data[j] for j in range(self.n_nodes)
            )  # check if all neighbors' have been received
            sync = False

            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(
                    0 == self.received_data[j][0][0] for j in range(self.n_nodes)
                )  # True if all True
            if sync:
                self.reorder_s_init(self.received_data)
                self.step += 1

        else:  # Have all messages at time t-1 arrived?
            # Check if lists are nonempty
            all_received = all(
                self.received_data[j] for j in range(self.n_nodes)
            )  # check if all neighbors' have been received
            sync = False

            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(
                    self.step - 1 == self.received_data[j][0][0] for j in range(self.n_nodes)
                )  # True if all True

            if sync:
                # Reorder the state vector received from the neighbors
                self.reorder_s_init(self.received_data)
                # save data on csv file
                # copy.deepcopy(self.s.tolist())
                # data_for_csv = [str(round(element, 4)) for element in data_for_csv[1:]]
                # data_for_csv = ','.join(data_for_csv)
                # writer(self.filename, data_for_csv + '\n')
                self.get_logger().info(f'Iter:{self.step}\n s:{self.s_history[0][-1]}')

                # Stop the node if tt exceeds MAXITERS
                if self.step > self.n_steps:
                    print('\nMAXITERS reached')
                    if st.simulation:
                        save_snapshots(
                            self.s_history,
                            self.goals,
                            None,
                            st.dt,
                            [(self.step - 1) * st.dt],
                            f'{self.out_dir}/snapshot',
                            x_lim=[-10, 10],
                            y_lim=[-8, 8],
                            flags=self.flags,
                        )

                        display_animation(
                            self.s_history,
                            self.goals,
                            None,
                            st.dt,
                            st.visual_method,
                            video_name=f'{self.out_dir}/video.mp4',
                            x_lim=[-10, 10],
                            y_lim=[-8, 8],
                            flags=self.flags,
                        )
                    else:
                        self.get_logger().info('My work is done, no plot requested. Goodbye!')

                    self.destroy_node()

                # update iteration counter
                self.step += 1

    def reorder_s_init(self, state_meas: list[float]):
        s = []
        for j in range(self.n_nodes):
            s_j = [s for s in state_meas[j].pop(0)[1:]]
            s.append(s_j)
        self.s_history.append([[], s])


def main(args=None):
    rclpy.init(args=args)

    agent = Agent()
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
