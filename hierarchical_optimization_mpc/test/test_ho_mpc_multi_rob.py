import traceback

import casadi as ca
import numpy as np

from hierarchical_optimization_mpc.ho_mpc_multi_robot import HOMPCMultiRobot, QPSolver
from hierarchical_optimization_mpc.tasks_creator_ho_mpc_mr import (
    TasksCreatorHOMPCMultiRobot,
)


class TestHOMPCMultiRobot:
    class HOMPCDesc:
        def __init__(
            self,
            states: list[ca.SX],
            inputs: list[ca.SX],
            fs: list[ca.SX],
            n_robots: list[int],
            hierarchical: bool,
            solver: QPSolver,
        ) -> None:
            self.states = states
            self.inputs = inputs
            self.fs = fs
            self.n_robots = n_robots
            self.hierarchical = hierarchical
            self.solver = solver

        def __repr__(self) -> str:
            return repr(f'n_robots = {self.n_robots}, hierarchical = {self.hierarchical}')

    def test_object_creation(self):
        try:
            self.hompcs_desc: list[self.HOMPCDesc] = []

            s = []
            u = []
            dt = 0.1  # timestep size
            s_kp1 = []

            s.append(ca.SX.sym('x', 3))  # state
            u.append(ca.SX.sym('u', 2))  # input

            # state_{k+1} = s_kpi(state_k, input_k)
            s_kp1.append(
                ca.vertcat(
                    s[0][0] + dt * u[0][0] * ca.cos(s[0][2] + 1 / 2 * dt * u[0][1]),
                    s[0][1] + dt * u[0][0] * ca.sin(s[0][2] + 1 / 2 * dt * u[0][1]),
                    s[0][2] + dt * u[0][1],
                )
            )

            s.append(ca.SX.sym('x2', 2))  # state
            u.append(ca.SX.sym('u2', 2))  # input

            # state_{k+1} = s_kpi(state_k, input_k)
            s_kp1.append(
                ca.vertcat(
                    s[1][0] + dt * u[1][0],
                    s[1][1] + dt * u[1][1],
                )
            )

            n_robots = [2]
            hierarchical = True
            self.hompcs_desc.append(
                self.HOMPCDesc(
                    s[0:1],
                    u[0:1],
                    s_kp1[0:1],
                    n_robots,
                    hierarchical,
                    QPSolver.quadprog,
                )
            )

            for n_robots in [[2, 0], [0, 2], [0, 2]]:
                hierarchical = True
                self.hompcs_desc.append(
                    self.HOMPCDesc(s, u, s_kp1, n_robots, hierarchical, QPSolver.quadprog)
                )

        except Exception as e:
            assert False, (
                f'HOMPCMultiRobot object creation with parameters '
                f'n_robots = {n_robots} and hierarchical = {hierarchical} '
                f'failed. It raised the exception {e}.'
            )

    def test_tasks_creator(self):
        dt = 0.01

        for hompc_desc in self.hompcs_desc:
            try:
                tasks_creator = TasksCreatorHOMPCMultiRobot(
                    hompc_desc.states,
                    hompc_desc.inputs,
                    hompc_desc.fs,
                    dt,
                    hompc_desc.n_robots,
                )

                tasks_creator.bounding_box = np.array([-20, 20, -20, 20])

                task_input_limits = tasks_creator.get_task_input_limits()

                task_input_smooth, task_input_smooth_coeffs = tasks_creator.get_task_input_smooth()

                task_centroid_vel_ref = tasks_creator.get_task_centroid_vel_ref([3, 1])

                task_vel_ref, task_vel_ref_coeff = tasks_creator.get_task_vel_ref([3, 1])

                task_coverage, task_coverage_coeff = tasks_creator.get_task_pos_ref(
                    [
                        [np.random.rand(2) for n_j in range(hompc_desc.n_robots[c])]
                        for c in range(len(hompc_desc.n_robots))
                    ]
                )

                task_charge, task_charge_coeff = tasks_creator.get_task_pos_ref(
                    [
                        [np.array([19, 19]) for n_j in range(hompc_desc.n_robots[c])]
                        for c in range(len(hompc_desc.n_robots))
                    ]
                )

                aux, mapping, task_formation, task_formation_coeff = (
                    tasks_creator.get_task_formation()
                )

                task_input_min = tasks_creator.get_task_input_min()

            except Exception:
                assert False, (
                    f'TasksCreatorHOMPCMultiRobot object creation with parameters '
                    f'n_robots = {hompc_desc.n_robots} and hierarchical = {hompc_desc.hierarchical} '
                    f'failed. It raised the following exception:\n {traceback.format_exc()}.'
                )
