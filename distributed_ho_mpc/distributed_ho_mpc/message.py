from dataclasses import dataclass
import numpy as np

@dataclass
class Message:
    sender_id: int
    x_i: np.ndarray
    x_j: np.ndarray
    rho_i: np.ndarray
    rho_j: np.ndarray

# TODO: support multiple priorities.
class MessageSender:
    """TODO"""
    
    def __init__(
        self,
        sender_id: int,
        adjacency_vector: np.ndarray,
        y: np.ndarray,
        rho: np.ndarray,
        n_xi: int
    ):
        self.sender_id = sender_id
        self.adjacency_vector = adjacency_vector
        
        self.y = y
        self.rho = rho
        
        self.n_xi = n_xi
        
    def send_message(self, receiver_id: int) -> Message:
        """TODO"""
        
        #! This assumes that all the x_i have the same size
        
        x_i = self.y[0:self.n_xi]
        rho_i = self.rho[0:self.n_xi]
        
        receiver_idx = list(self.adjacency_vector).index(receiver_id)
        x_j = self.y[receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi]
        rho_j = self.rho[receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi]
        
        return Message(self.sender_id, x_i, x_j, rho_i, rho_j)
    