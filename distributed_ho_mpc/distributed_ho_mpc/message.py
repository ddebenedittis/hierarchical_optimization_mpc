from dataclasses import dataclass
import numpy as np

@dataclass
class Message:
    sender_id: int
    x_i: np.ndarray
    x_j: np.ndarray
    rho_i: np.ndarray
    rho_j: np.ndarray
    message: str 

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

        
        
    def send_message(self, receiver_id: int, message) -> Message:
        """TODO"""
        
        #! This assumes that all the x_i have the same size
        receiver_idx = list(self.adjacency_vector).index(receiver_id)

        if message == 'P': 
            x_i = self.y[0:self.n_xi]
            x_j = self.y[receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi]

            return Message(self.sender_id, x_i, x_j, rho_i=None, rho_j=None, message='P')
        
        if message == 'D':
            rho_i = self.rho[0:self.n_xi]
            rho_j = self.rho[receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi]
            
            return Message(self.sender_id,  x_i=None, x_j=None, rho_i=rho_i, rho_j=rho_j, message='D')
        
        

class MessageReceiver:
    """TODO"""
    
    def __init__(
            self, 
            receiver_id: int,
            neighbour_order: list
        ):

        self.receiver_id = receiver_id
        self.neighbour_order = neighbour_order        
        self.messages = []
        
    def receive_message(self, message: Message):
        if message.sender_id == self.receiver_id:
            return
        
        self.messages.append(message)