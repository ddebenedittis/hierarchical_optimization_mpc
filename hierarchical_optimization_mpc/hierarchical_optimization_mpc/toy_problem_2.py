import copy
import sys

import casadi as ca
import numpy as np
import progressbar as pb

from hierarchical_optimization_mpc.auxiliary.evolve import evolve
from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, QPSolver, TaskType
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import TasksCreatorHOMPCMultiRobot
from hierarchical_optimization_mpc.utils.disp_het_multi_rob import display_animation
from hierarchical_optimization_mpc.utils.disp_het_multi_rob_toy_2 import save_snapshots
from hierarchical_optimization_mpc.utils.robot_models import get_omnidirectional_model, get_unicycle_model


np.set_printoptions(
    precision=4,
    linewidth=300,
    suppress=True,
    threshold=sys.maxsize,
)


def exp(
    initial_state,
    hierarchical = True, solver = QPSolver.quadprog, visual_method = 'plot',
    weights: list[int] | None = None,
    s_histories = [],
):
    
    np.random.seed(0)
    
    dt = 0.1
    
    n_robots = [1, 0]
    
    # ============================= System Model ============================= #
    
    s = [None for _ in range(2)]
    u = [None for _ in range(2)]
    s_kp1 = [None for _ in range(2)]
    
    s[0], u[0], s_kp1[0] = get_unicycle_model(dt)
    s[1], u[1], s_kp1[1] = get_omnidirectional_model(dt)
    
    # =========================== Define The Tasks =========================== #
    
    goals = [
        np.array([5, 10]),
    ]
    
    tasks_creator = TasksCreatorHOMPCMultiRobot(
        s, u, s_kp1, dt, n_robots,
    )
    
    tasks_creator.bounding_box = np.array([-20, 20, -20, 20])
    
    task_input_limits = tasks_creator.get_task_input_limits()
    
    task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()
    
    task_x_coord_1 = [
        ca.vertcat(s_kp1[0][0]) - 5,
        ca.vertcat(s_kp1[1][0]),
    ]
    
    task_x_coord_2 = [
        ca.vertcat(s_kp1[0][0]) + 12,
        ca.vertcat(s_kp1[1][0]),
    ]
    
    task_y_coord_1 = [
        ca.vertcat(s_kp1[0][1]) - 10,
        ca.vertcat(s_kp1[1][1]),
    ]
    
    task_y_coord_2 = [
        ca.vertcat(s_kp1[0][1]) + 4,
        ca.vertcat(s_kp1[1][1]),
    ]
    
    task_theta_coord_1 = [
        ca.vertcat(s_kp1[0][2]) - 1/2*np.pi/2,
        ca.vertcat(s_kp1[1][0]),
    ]
    
    task_theta_coord_2 = [
        ca.vertcat(s_kp1[0][2]) + 1/4*np.pi/2,
        ca.vertcat(s_kp1[1][0]),
    ]
    
    task_input_min = tasks_creator.get_task_input_min()
    
    # ============================ Create The MPC ============================ #
    
    hompc = HOMPCMultiRobot(
        s, u, s_kp1, n_robots, solver = solver,
        hierarchical=hierarchical,
    )
    hompc.n_control = 4
    hompc.n_pred = 0
    
    hompc.create_task(
        name = "input_limits", prio = 1,
        type = TaskType.Same,
        ineq_task_ls = task_input_limits,
        # ineq_task_coeff = task_input_limits_coeffs,
        ineq_weight = weights[0] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "input_smooth", prio = 2,
        type = TaskType.SameTimeDiff,
        ineq_task_ls = task_input_smooth,
        ineq_task_coeff = task_input_smooth_coeffs,
        ineq_weight = weights[1] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "x_coord_1", prio = 3,
        type = TaskType.Same,
        eq_task_ls = task_x_coord_1,
        # eq_task_coeff = task_pos_coeff[0],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[2] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "x_coord_2", prio = 4,
        type = TaskType.Same,
        eq_task_ls = task_x_coord_2,
        # eq_task_coeff = task_pos_coeff[0],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[3] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "y_coord_1", prio = 5,
        type = TaskType.Same,
        eq_task_ls = task_y_coord_1,
        # eq_task_coeff = task_pos_coeff[1],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[4] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "y_coord_2", prio = 6,
        type = TaskType.Same,
        eq_task_ls = task_y_coord_2,
        # eq_task_coeff = task_pos_coeff[1],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[5] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "theta_coord_1", prio = 7,
        type = TaskType.Same,
        eq_task_ls = task_theta_coord_1,
        # eq_task_coeff = task_pos_coeff[2],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[6] if weights is not None else 1.0,
    )
    
    hompc.create_task(
        name = "theta_coord_2", prio = 8,
        type = TaskType.Same,
        eq_task_ls = task_theta_coord_2,
        # eq_task_coeff = task_pos_coeff[2],
        time_index = [0, 1, 2, 3],
        eq_weight = weights[7] if weights is not None else 1.0,
    )
    
    # ======================================================================== #
    
    s = copy.deepcopy(initial_state)
    
    n_steps = 200
    
    s_history = [None] * n_steps
    
    time_to_goal = None
        
    for k in pb.progressbar(range(n_steps)):
        u_star = hompc(copy.deepcopy(s))
        
        s = evolve(s, u_star, dt)
        
        if time_to_goal is None:
            threshold = 1.
            if np.linalg.norm(s[0][0][0:2] - goals[0]) < threshold:
                d_prev = np.linalg.norm(s_history[k-1][0][0][0:2] - goals[0])
                d_now = np.linalg.norm(s[0][0][0:2] - goals[0])
                temp = (d_prev - threshold) / (d_prev - d_now)
                
                time_to_goal = (k-1) * dt + temp * dt
        
        s_history[k] = copy.deepcopy(s)
        
    print(s)
    
    if time_to_goal is not None:
        print(f"Time to reach the goal: {time_to_goal:.2f} s")
    else:
        print("The goal was not reached.")
    
    print( "The time was used in the following phases:")
    max_key_len = max(map(len, hompc.solve_times.keys()))
    for key, value in hompc.solve_times.items():
        key_len = len(key)
        print(f"{key}: {' '*(max_key_len-key_len)}{value}")
    
    # if visual_method is not None and visual_method != 'none':
    #     display_animation(s_history, goals, None, dt, visual_method)
        
    if visual_method == 'save':
        save_snapshots(s_histories + [s_history], goals, dt, [0, 10], 'snapshot')
        
    return s_history
        

def main():
    initial_state = [
        [np.array([0, 0, 0])],
        [],
    ]
    visual_method = 'save'
    
    s_histories = []
    
    s_histories.append(exp(initial_state=initial_state, hierarchical=True,
        visual_method='none', s_histories=s_histories))
    
    kappa = 3**-1
    weights = [100.0, 100.0] + [kappa**i for i in range(6)]
    s_histories.append(exp(initial_state=initial_state, hierarchical=False,
       visual_method='none', weights=weights, s_histories=s_histories))
    
    kappa = 10**-2
    weights = [100.0, 100.0] + [kappa**i for i in range(6)]
    s_histories.append(exp(initial_state=initial_state, hierarchical=False,
       visual_method=visual_method, weights=weights, s_histories=s_histories))
            
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('--solver',
        metavar="{clarabel, osqp, proxqp, quadprog, reluqp}", default='quadprog', required=False,
        help='QP solver to use'
    )
    parser.add_argument('--visual_method',
        metavar="{plot, save, none}", default='none', required=False,
        help='How to display the results'
    )
    args = parser.parse_args()
    
    initial_state = [
        [np.array([0, 0, 0])],
        [],
    ]
    
    print("Hierarchical\n")
    exp(
        initial_state=initial_state,
        hierarchical=True,
        solver=args.solver,
        visual_method=args.visual_method,
    )
    print("\n")
    
    print("Weighted - kappa = 5\n")
    kappa = 3**-1
    weights = [100.0, 100.0] + [kappa**i for i in range(6)]
    exp(
        initial_state=initial_state,
        hierarchical=False,
        solver=args.solver,
        visual_method=args.visual_method,
        weights=weights,
    )
    print("\n")
    
    print("Weighted - kappa = 100\n")
    kappa = 10**-2
    weights = [100.0, 100.0] + [kappa**i for i in range(6)]
    exp(
        initial_state=initial_state,
        hierarchical=False,
        solver=args.solver,
        visual_method=args.visual_method,
        weights=weights,
    )
    print("\n")
