#!/usr/bin/env python3

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from hierarchical_optimization_mpc.ho_mpc import HOMPC

np.set_printoptions(precision=3, linewidth=300, suppress=True)


class Animation:
    def __init__(self, scat, data) -> None:
        self.scat = scat
        self.data = data

    def update(self, frame):
        self.scat.set_offsets((self.data[frame, 0], self.data[frame, 1]))
        return self.scat


def main():
    # ======================== Define The System Model ======================= #

    # Define the state and input variables, and the discrete-time dynamics model.

    s = ca.SX.sym('x', 3)  # state
    u = ca.SX.sym('u', 2)  # input
    dt = 0.01  # timestep size

    # state_{k+1} = s_kpi(state_k, input_k)
    s_kp1 = ca.vertcat(
        s[0] + dt * u[0] * ca.cos(s[2]), s[1] + dt * u[0] * ca.sin(s[2]), s[2] + dt * u[1]
    )

    # =========================== Define The Tasks =========================== #

    # Define the tasks separately.

    # Input limits
    v_max = 5
    v_min = -5
    omega_max = 1
    omega_min = -1

    task_input_limits = ca.vertcat(u[0] - v_max, -u[0] + v_min, u[1] - omega_max, -u[1] + omega_min)
    task_input_limits_coeffs = [np.array([0, 0, 0, 0])]

    task_vel_reference = ca.vertcat((s_kp1[0] - s[0]) / dt - 1, (s_kp1[1] - s[1]) / dt - 0)

    task_input_min = u

    # ============================ Create The MPC ============================ #

    hompc = HOMPC(s, u, s_kp1)
    hompc.n_control = 3
    hompc.n_pred = 0

    hompc.create_task(
        name='input_limits',
        prio=1,
        ineq_task_ls=task_input_limits,
        ineq_task_coeff=task_input_limits_coeffs,
    )

    hompc.create_task(
        name='reference',
        prio=2,
        eq_task_ls=task_vel_reference,
    )

    hompc.create_task(
        name='input_minimization',
        prio=3,
        eq_task_ls=task_input_min,
    )

    # ======================================================================== #

    s = np.array([0.0, 0.0, 0.8])

    n_steps = 1000

    s_history = np.zeros((n_steps, 3))

    for k in range(n_steps):
        print(k)

        u_star = hompc(s)

        for i in range(10):
            s = s + dt / 10 * np.array(
                [u_star[0] * np.cos(s[2]), u_star[0] * np.sin(s[2]), u_star[1]]
            )

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
