from dataclasses import dataclass

@dataclass
class Message:
    """Representing the message with primal variables that should be passed between nodes. This class is a simple data-holder which
    contains the required data to be transferred between nodes. """

    node_id: int    # Identifier of the agent that send the message
    s: list # state vector
    u: list
    x: list # s+u
    Z: dict # Nullspace 
    #mapping: list # mapping of the opt_vector: ex. [(0,8), (1,8), (2,8)] means that the node has states of 0, 1 and 2
    
@dataclass
class Message_dual:
    """Representing the message with dual variables that should be passed between nodes. This class is a simple data-holder which
    contains the required dual variables to be transferred between nodes. """
    
    node_id: int    # Identifier of the agent that send the message
    rho_j: list     # dual variable