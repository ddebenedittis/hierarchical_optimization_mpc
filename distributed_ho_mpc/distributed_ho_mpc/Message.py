from dataclasses import dataclass
import numpy as np

@dataclass
class Message:
    """Representing the message that should be passed between nodes. This class is a simple data-holder which
    contains the required data to be transferred between nodes. """

    node_id: int    # Identifier of the agent that send the message
    node_xi: float  # Shared information
    s: list # state vector
    u: list
    Z : dict # Nullspace
    Xsym : list # symbolic component of the optimization vector 
    
