import casadi as ca
import numpy as np


class HOMPCValidator:
    @staticmethod
    def validate_task_ls(task_ls):
        if task_ls is None:
            return

        if not isinstance(task_ls, list):
            raise ValueError('task_ls must be a list.')

        for e in task_ls:
            if not isinstance(e, ca.SX):
                raise ValueError('Each element in task_ls must be a casadi SX.')

    @staticmethod
    def validate_task_coeff(task_coeff):
        if task_coeff is None:
            return

        desc = """\n
            task_coeff is a list[list[list[np.ndarray]]]. The three levels
            iterate over the robot classes c, the robot indices j, and the
            timesteps k, respectively. I.e. task_coeff[c][j][k].
        """

        if not isinstance(task_coeff, list):
            raise ValueError('task_coeff must be a list of lists of lists of np.ndarrays.' + desc)

        for e1 in task_coeff:
            if not isinstance(e1, list):
                raise ValueError(
                    'task_coeff must be a list of lists of lists of np.ndarrays.' + desc
                )

            for e2 in e1:
                if not isinstance(e2, list):
                    raise ValueError(
                        'task_coeff must be a list of lists of lists of np.ndarrays.' + desc
                    )

                for e3 in e2:
                    if not isinstance(e3, np.ndarray):
                        raise ValueError(
                            'task_coeff must be a list of lists of lists of np.ndarrays.' + desc
                        )
