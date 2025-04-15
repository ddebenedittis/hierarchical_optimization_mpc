from dataclasses import dataclass

@dataclass
class Message:
    """Representing the message with primal variables that should be passed between nodes. This class is a simple data-holder which
    contains the required data to be transferred between nodes. """

    node_id: int    # Identifier of the agent that send the message
    x_i: list # state vector
    x_j: list # s+u
    
@dataclass
class Message_dual:
    """Representing the message with dual variables that should be passed between nodes. This class is a simple data-holder which
    contains the required dual variables to be transferred between nodes. """
    
    node_id: int    # Identifier of the agent that send the message
    rho_j: list     # dual variable
    j: int          # id of the node to send the message