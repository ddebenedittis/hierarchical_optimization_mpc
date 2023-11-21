#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
from casadi import *
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from hompc_approximator.nn_regressor import Model


np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True
)


class Animation():
    def __init__(self, scat, data) -> None:
        self.scat = scat
        self.data = data
    
    def update(self, frame):
        self.scat.set_offsets((self.data[frame, 0], self.data[frame, 1]))
        return self.scat
    
def process_data(unprocessed_data):
    processed_data = np.zeros(len(unprocessed_data) + 1)
    # processed_data[:, 0:2] = unprocessed_data[:, 0:2]
    processed_data[2] = np.cos(unprocessed_data[2])
    processed_data[3] = np.sin(unprocessed_data[2])
    
    return processed_data


def main():
    pkg_share_dir = get_package_share_directory('hompc_approximator')
    model = Model(4, 50, 2)
    model.load_state_dict(torch.load(pkg_share_dir + '/nn/model.zip'))
    print(model.eval())
    
    dt = 0.01
    
    # ======================================================================== #
    
    s = np.array([0., 0., 0.8])
    
    n_steps = 1000
    
    s_history = np.zeros((n_steps, 3))
        
    for k in range(n_steps):
        print(k)
        
        u_star = np.array([
            model.forward(
                torch.tensor(process_data(s)).to(torch.float32)
            ).detach().numpy()
        ]).flatten()
        print(u_star)
                
        for i in range(10):
            s = s + dt / 10 * np.array([
                u_star[0] * np.cos(s[2]),
                u_star[0] * np.sin(s[2]),
                u_star[1]
            ])
        
        print(s)
        print(u_star)
        print()
        
        s_history[k, :] = s
        
    fig, ax = plt.subplots()
    scat = ax.scatter(s[0], s[1])
    ax.set(xlim=[-5, 5], ylim=[-5, 5], xlabel='x [m]', ylabel='y [m]')
    ax.legend()
    
    anim = Animation(scat, s_history)
    
    temp = FuncAnimation(fig=fig, func=anim.update, frames=range(1000), interval=30)
    plt.show()
    
    
if __name__ == '__main__':
    main()
