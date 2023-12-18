#!/usr/bin/env python3

import copy
from dataclasses import dataclass
import sys

import casadi as ca
from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, TaskType
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(
    precision=3,
    linewidth=300,
    suppress=True,
    threshold=sys.maxsize,
)
    

def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO,mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale


def evolve(s, u_star, dt):
    n_intervals = 10
    
    for c in range(len(s)):
        for j in range(len(s[c])):
            if c == 0:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array([
                        u_star[c][j][0] * np.cos(s[c][j][2]),
                        u_star[c][j][0] * np.sin(s[c][j][2]),
                        u_star[c][j][1],
                    ])
            
            if c == 1:
                for _ in range(n_intervals):
                    s[c][j] = s[c][j] + dt / n_intervals * np.array([
                        u_star[c][j][0],
                        u_star[c][j][1],
                    ])
                    
    return s


@dataclass
class MultiRobotScatter:
    unicycles: ...
    omnidir: ...
    centroid: ...


class Animation():
    def __init__(self, scat: MultiRobotScatter, data) -> None:
        self.scat = scat
        self.data = data
        
        self.n_robots = [len(data_i) for data_i in data[0]]
    
    def update(self, frame):
        state = self.data[frame]
        
        x = [
            np.zeros((self.n_robots[0], 3)),
            np.zeros((self.n_robots[1], 2)),
        ]
        
        for c in range(len(state)):
            for j, s_c_j in enumerate(state[c]):
                x[c][j, 0] = s_c_j[0]
                x[c][j, 1] = s_c_j[1]
                if c == 0:
                    x[c][j, 2] = s_c_j[2]
                
        for i in range(self.n_robots[0]):
            self.scat.unicycles[i].remove()
        for i in range(self.n_robots[1]):
            self.scat.omnidir[i].remove()
                
        for i in range(self.n_robots[0]):
            deg = x[0][i,2] * 180 / np.pi
            marker, scale = gen_arrow_head_marker(deg)
                        
            self.scat.unicycles[i] = plt.scatter(
                x = x[0][i,0], y = x[0][i,1],
                s = 250 * scale**2, c = 'C0',
                marker = marker,
            )
            
        for i in range(self.n_robots[1]):
            self.scat.omnidir[i] = plt.scatter(
                x = x[1][i,0], y = x[1][i,1],
                s = 25, c = 'C1',
                marker = 'o',
            )
        
        self.scat.centroid.set_offsets(
            (np.mean(x[0][:,0:2],axis=0)*self.n_robots[0] + np.mean(x[1][:,0:2],axis=0)*self.n_robots[1]) / sum(self.n_robots)
        )
        
        return self.scat


def main():
    np.random.seed()
    
    # ======================== Define The System Model ======================= #
    
    # Define the state and input variables, and the discrete-time dynamics model.
    
    s = []
    u = []
    dt = 0.01           # timestep size
    s_kp1 = []
    
    s.append(ca.SX.sym('x', 3))     # state
    u.append(ca.SX.sym('u', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[0][0] + dt * u[0][0] * ca.cos(s[0][2] + 1/2*dt * u[0][1]),
        s[0][1] + dt * u[0][0] * ca.sin(s[0][2] + 1/2*dt * u[0][1]),
        s[0][2] + dt * u[0][1]
    ))
    
    
    s.append(ca.SX.sym('x2', 2))     # state
    u.append(ca.SX.sym('u2', 2))     # input
    
    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1.append(ca.vertcat(
        s[1][0] + dt * u[1][0],
        s[1][1] + dt * u[1][1],
    ))
    
    n_robots = [2, 1]
    
    # =========================== Define The Tasks =========================== #
    
    # Input limits
    v_max = 5
    v_min = -5
    omega_max = 1
    omega_min = -1
    
    task_input_limits = [
        ca.vertcat(
            u[0][0] - v_max,
            - u[0][0] + v_min,
            u[0][1] - omega_max,
            - u[0][1] + omega_min
        ),
        ca.vertcat(
            u[1][0] - v_max,
            - u[1][0] + v_min,
            u[1][1] - omega_max,
            - u[1][1] + omega_min
        ),
    ]
    
    task_input_limits_coeffs = [
        [np.array([0, 0, 0, 0])],
        [np.array([0, 0, 0, 0])],
    ]
    
    task_input_smooth = [
        ca.vertcat(
            u[0][0],
            - u[0][0],
            u[0][1],
            - u[0][1]
        ),
        ca.vertcat(
            u[1][0],
            - u[1][0],
            u[1][1],
            - u[1][1]
        ),
    ]
    
    task_input_smooth_coeffs = [
        [np.array([0.9, 0.9, 0.8, 0.8])],
        [np.array([0.9, 0.9, 0.8, 0.8])],
    ]
    
    # Velocity reference
    task_vel_ref = [
        ca.vertcat(
            (s_kp1[0][0] - s[0][0]) / dt - 1,
            (s_kp1[0][1] - s[0][1]) / dt - 0,
        ) / sum(n_robots),
        ca.vertcat(
            (s_kp1[1][0] - s[1][0]) / dt - 1,
            (s_kp1[1][1] - s[1][1]) / dt - 0,
        ) / sum(n_robots),
    ]
    
    task_input_min = [
        ca.vertcat(
            u[0][0],
            u[0][1],
        ),
        ca.vertcat(
            u[1][0],
            - u[1][0],
            u[1][1],
            - u[1][1]
        ),
    ]
    
    aux = ca.SX.sym('aux', 2, 2)
    
    mapping = [
        ca.vertcat(
            s[0][0],
            s[0][1],
        ),
        ca.vertcat(
            s[1][0],
            s[1][1],
        ),
    ]
    
    task_avoid_collision = ca.vertcat(
        (aux[0,0] - aux[1,0])**2 + (aux[0,1] - aux[1,1])**2 - 5,
    )
    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(s, u, s_kp1, n_robots)
    hompc.n_control = 3
    hompc.n_pred = 0
    
    hompc.create_task(
        name = "input_limits", prio = 1,
        type = TaskType.Same,
        ineq_task_ls = task_input_limits,
        ineq_task_coeff = task_input_limits_coeffs,
    )
    
    hompc.create_task(
        name = "input_smooth", prio = 2,
        type = TaskType.SameTimeDiff,
        ineq_task_ls = task_input_smooth,
        ineq_task_coeff = task_input_smooth_coeffs,
    )
    
    # hompc.create_task_bi(
    #     name = "input_smooth", prio = 3,
    #     type = TaskType.Bi,
    #     aux = aux,
    #     mapping = mapping,
    #     eq_task_ls = task_avoid_collision,
    # )
    
    hompc.create_task(
        name = "vel_ref", prio = 3,
        type = TaskType.Sum,
        eq_task_ls = task_vel_ref,
    )
    
    hompc.create_task(
        name = "input_minimization", prio = 4,
        type = TaskType.Same,
        eq_task_ls = task_input_min,
    )
    
    # ======================================================================== #
    
    s = [
        [np.multiply(np.random.random((3)), np.array([2, 2, 2*np.pi])) + np.array([-1, -1, 0])
         for _ in range(n_robots[0])],
        [np.multiply(np.random.random((2)), np.array([2, 2])) + np.array([-1, -1])
         for _ in range(n_robots[1])],
    ]
    
    n_steps = 1000
    
    s_history = [None] * n_steps
        
    for k in range(n_steps):
        print(k)
                
        u_star = hompc(copy.deepcopy(s))
        
        print(s)
        print(u_star)
        print()
                    
        s = evolve(s, u_star, dt)
                
        s_history[k] = copy.deepcopy(s)
            
    fig, ax = plt.subplots()
    x = [np.zeros(n_r) for n_r in n_robots]
    y = [np.zeros(n_r) for n_r in n_robots]
    for c, n_r in enumerate(n_robots):
        for j in range(n_r):
            state = s[c][j]
            x[c][j] = state[0]
            y[c][j] = state[1]
    
    scat = MultiRobotScatter
    scat.unicycles = [None] * n_robots[0]
    scat.omnidir = [None] * n_robots[1]
        
    for i in range(n_robots[0]):
        scat.unicycles[i] = ax.scatter(x[0], y[0], 25, 'C0')
    for i in range(n_robots[1]):
        scat.omnidir[i] = ax.scatter(x[1], y[1], 25, 'C1')
        
    scat.centroid = ax.scatter(
        (np.mean(x[0])*n_robots[0] + np.mean(x[1])*n_robots[1]) / sum(n_robots),
        (np.mean(y[0])*n_robots[0] + np.mean(y[1])*n_robots[1]) / sum(n_robots),
        25, 'C2')
    
    ax.set(xlim=[-10., 10.], ylim=[-10., 10.], xlabel='x [m]', ylabel='y [m]')
    
    marker, scale = gen_arrow_head_marker(0)
    legend_elements = [
        Line2D([], [], marker=marker, markersize=20*scale, color='C0', linestyle='None', label='Unicycles'),
        Line2D([], [], marker='o', color='C1', linestyle='None', label='Omnidirectional Robot'),
        Line2D([], [], marker='o', color='C2', linestyle='None', label='Fleet Centroid'),
    ]
    
    ax.legend(handles=legend_elements)
    
    anim = Animation(scat, s_history)
    
    temp = FuncAnimation(fig=fig, func=anim.update, frames=range(n_steps), interval=30)
    plt.show()
    
    
if __name__ == '__main__':
    try:
        main()
    except:
        pass
