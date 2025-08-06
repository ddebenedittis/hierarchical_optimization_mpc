from dataclasses import dataclass
from enum import Enum
from typing import Any

import casadi as ca
import numpy as np


class RobotType(Enum):
    OMNIDIRECTIONAL = 1
    UNICYCLE = 2


@dataclass
class RobCont:
    """
    Generic container to hold quantities related to multi-robot system.
    Each attribute is associated with a specific robot type.
    """

    omni: Any = None
    uni: Any = None

    def tolist(self) -> list:
        """Convert the class to a list of its attributes."""

        return [[self.omni], [self.uni]]

    def tolist_model(self, model: str = 'omni') -> list:
        """
        Convert the class to a list of its attributes for a specific model.

        Args:
            model (str): 'omni' for omnidirectional, 'uni' for unicycle
        Returns:
            list: list of attributes for the specified model
        """
        if model == 'omni':
            return [self.omni]
        elif model == 'uni':
            return [self.uni]
        else:
            raise ValueError(f"Model {model} not recognized. Use 'omni' or 'uni'.")

    def expand(self, state_meas, model: str = 'omniwheel'):
        """
        Expand the container to hold multiple robots of the same type.

        Args:
            state_meas (Any): state measurement to be added
            model (str): 'omni' for omnidirectional, 'uni' for un
        """
        if model == 'omniwheel':
            self.omni.append(state_meas)
        elif model == 'unicycle':
            self.uni.append(state_meas)
        else:
            raise ValueError(f"Model {model} not recognized. Use 'omni' or 'uni'.")

    def reduce(self, idx: int, model: str = 'omniwheel'):
        """
        Reduce the container to hold only the robot that still have a connection.

        Args:
            index (int): index of the robot to remove
            model (str): 'omni' for omnidirectional, 'uni' for unicycle
        """
        if model == 'unicycle':
            if self.uni is not None:
                self.uni.pop(idx)
            else:
                raise ValueError('RobCont is empty, cannot reduce.')
        if model == 'omniwheel':
            if self.omni is not None:
                self.omni.pop(idx)
            else:
                raise ValueError('RobCont is empty, cannot reduce.')


def get_unicycle_model(dt: float):
    """
    Get the symbolic state and input variables and the discrete-time dynamics
    model for the unicycle model.

    Args:
        dt (float): sample time

    Returns:
        state, input, next_state
    """

    s = ca.SX.sym('s_unicycle', 3)
    u = ca.SX.sym('u_unicycle', 2)

    x = s[0]
    y = s[1]
    theta = s[2]

    v = u[0]
    omega = u[1]

    s_kp1 = ca.vertcat(
        x + dt * ca.cos(theta + 1 / 2 * dt * omega) * v,
        y + dt * ca.sin(theta + 1 / 2 * dt * omega) * v,
        theta + dt * omega,
    )

    return s, u, s_kp1


def get_omnidirectional_model(dt: float):
    """
    Get the symbolic state and input variables and the discrete-time dynamics
    model for the omnidirectional model.
    Args:
        dt (float): sample time

    Returns:
        state, input, next_state
    """

    s = ca.SX.sym('s_omni', 2)
    u = ca.SX.sym('u_omni', 2)

    x = s[0]
    y = s[1]

    v_x = u[0]
    v_y = u[1]

    s_kp1 = ca.vertcat(
        x + dt * v_x,
        y + dt * v_y,
    )

    return s, u, s_kp1
