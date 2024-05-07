import io
from contextlib import redirect_stdout
import datetime
import os

import pandas as pd

from hierarchical_optimization_mpc.example_multi_robot import main as ex_multi_robot_main
# from hierarchical_optimization_mpc.ho_mpc_multi_robot import QPSolver

n_robots_list = [
    [1,0],
    [0,2],
    [4,0],
    [0,6],
    [4,4],
    [5,5],
    [12,0],
    [7,7],
]

solvers = [
    'clarabel',
    'osqp',
    'proxqp',
    'quadprog',
]

methods = {
    'hierarchical': True,
    'weighted': False,
}

def compare_qp_solvers():
    times = {}
    
    for solver in solvers:
        times[solver] = {}
        print(f"\nRunning code for solver = {solver}.\n")
        for n_robots in n_robots_list:
            trap = io.StringIO()
            print(f"Running code for n_robots = {n_robots}.")
            with redirect_stdout(trap):
                time = ex_multi_robot_main(n_robots=n_robots, solver=solver, visual_method='none')
                
            times[solver][f"[{n_robots[0]},{n_robots[1]}]"] = time
            
    print(times)
    
    path = 'csv/'
    filename = f"qp_solvers_solve_times_{datetime.datetime.now():%Y-%m-%d-%H:%M:%S}.csv"
    
    try:
        os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed.")
    else:
        print(f"Successfully created the directory {path}.")
            
    df = pd.DataFrame(times)
    out_file = open(path+filename, 'wb')
    df.to_csv(out_file)
    out_file.close()
    
    
def compare_method_solve_times():
    times = {}
    
    for method in methods:
        times[method] = {}
        print(f"\nRunning code for solver = {method}.\n")
        for n_robots in n_robots_list:
            trap = io.StringIO()
            print(f"Running code for n_robots = {n_robots}.")
            with redirect_stdout(trap):
                time = ex_multi_robot_main(n_robots=n_robots, hierarchical=methods[method], visual_method='none')
                
            times[method][f"[{n_robots[0]},{n_robots[1]}]"] = time
            
    print(times)
    
    path = 'csv/'
    filename = f"method_solve_times_{datetime.datetime.now():%Y-%m-%d-%H:%M:%S}.csv"
    
    try:
        os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed.")
    else:
        print(f"Successfully created the directory {path}.")
            
    df = pd.DataFrame(times)
    out_file = open(path+filename, 'wb')
    df.to_csv(out_file)
    out_file.close()
    
    
def compare_all_solve_times():
    times = []
    
    for n_robots in n_robots_list:
        print(f"Running code for n_robots = {n_robots}.")
        for solver in solvers:
            print(f"\nRunning code for solver = {solver}.\n")
            for method in methods.keys():
                trap = io.StringIO()
                with redirect_stdout(trap):
                    time = ex_multi_robot_main(
                        n_robots=n_robots, solver=solver, visual_method='none',
                        hierarchical=methods[method],
                    )
                    
                times.append(
                    [f"[{n_robots[0]},{n_robots[1]}]", solver, method, time,]
                )
            
    print(times)
    
    path = 'csv/'
    filename = f"all_solve_times_{datetime.datetime.now():%Y-%m-%d-%H:%M:%S}.csv"
    
    
    try:
        os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed.")
    else:
        print(f"Successfully created the directory {path}.")
            
    df = pd.DataFrame(times, columns=['N Robots', 'Solver', 'Method', 'Time'])
    out_file = open(path+filename, 'wb')
    df.to_csv(out_file)
    out_file.close()
    
    
def main():
    compare_qp_solvers()
    compare_method_solve_times()
    # compare_all_solve_times()

if __name__ == '__main__':
    main()
