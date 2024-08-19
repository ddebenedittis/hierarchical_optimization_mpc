import casadi as ca

from hierarchical_qp.hierarchical_qp import QPSolver
from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot



class DistributedHOMPC():
    """"""
    
    def __init__(
        self, states: list[ca.SX], inputs: list[ca.SX], fs: list[ca.SX], n_robots: list[int],
        solver: QPSolver = QPSolver.quadprog,
        hierarchical = True,
    ) -> None:
        """
        Initialize the instance.

        Args:
            states (list[ca.SX]): list of state symbolic variable for each robot class
            inputs (list[ca.SX]): list of input symbolic variable for each robot class
            fs (list[ca.SX]):     list of discrete-time system equations for each robot class:
                                  state_{k+1} = f(state_k, input_k)
            n_robots(list[int]):  list of the number of robots for each robot class
        """
        
        self.n_robots = n_robots
        
        self.MPCs = []
