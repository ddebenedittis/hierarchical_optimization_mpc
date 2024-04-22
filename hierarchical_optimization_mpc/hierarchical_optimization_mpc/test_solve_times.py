import io
from contextlib import redirect_stdout
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
    'quadprog',
]

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
    filename = 'qp_solvers_solve_times.csv'
    
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
    
    
def main():
    compare_qp_solvers()


if __name__ == '__main__':
    main()
