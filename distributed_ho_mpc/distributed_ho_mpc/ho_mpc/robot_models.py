from dataclasses import dataclass
from enum import Enum
from typing import Any
import numpy as np
import casadi as ca


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
    
    def tolist(self) -> list:
        """Convert the class to a list of its attributes."""
        return [self.omni]
    
    def expand(self, state_meas):
        """
        Expand the container to hold multiple robots of the same type.
        Args:
            n (int): number of robots
        """
        #for s in state_meas:
        self.omni.append(state_meas)
    def reduce(self, idx: int):
        """
        Reduce the container to hold only the robot that still have a connection.
        Args:
            index (int): index of the robot to remove
        """
        if self.omni is not None:
            self.omni.pop(idx)
        else:
            raise ValueError("RobCont is empty, cannot reduce.")


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
        x + dt * ca.cos(theta + 1/2*dt*omega) * v,
        y + dt * ca.sin(theta + 1/2*dt*omega) * v,
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
