import casadi as ca



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
