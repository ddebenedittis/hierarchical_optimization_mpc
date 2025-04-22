from dataclasses import dataclass
import numpy as np

@dataclass
class Message:
    sender_id: int      # node id
    x_i: np.ndarray     # primal variable of the sender i
    x_j: np.ndarray     # primal variable of the neighbour j estimated by the sender i
    rho_i: np.ndarray   # dual variable of the sender
    rho_j: np.ndarray   # dual variable of the neighbour j estimated by the sender i
    update: str         # update type: 'P' for primal, 'D' for dual

# TODO: support multiple priorities.
class MessageSender:
    """
        Handle the communication phase between agents. Agents initialize this class and use its methods to send messages to their neighbours.
        NOTE! the update of y and rho must be perfmormed directly in the node, invoking the class variable
    """
    
    def __init__(
        self,
        sender_id: int,
        adjacency_vector: np.ndarray,
        y: np.ndarray,
        rho: np.ndarray,
        n_xi: int,
        n_priorities: int,
    ):
        self.sender_id = sender_id
        self.adjacency_vector = adjacency_vector
        self.y = y
        self.rho = rho
        self.n_xi = n_xi
        self.n_priorities = n_priorities
                
        
    def send_message(self, receiver_id: int, update: str) -> Message:
        """ 
            Divide the primal or dual variable in the correct information to send to the receiver agent.
            receiver_id: id of the receiver agent
            update: 'P' for primal, 'D' for dual
        """
        
        #! This assumes that all the x_i have the same size
        receiver_idx = list(self.adjacency_vector).index(receiver_id)

        if update == 'P':
            x_i = self.y[:, 0:self.n_xi]
            x_j = self.y[:, receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi]

            return Message(self.sender_id, x_i, x_j, rho_i=None, rho_j=None, update='P')
        
        if update == 'D':
            rho_i = self.rho[0, :, (receiver_idx -1) * self.n_xi: (receiver_idx) * self.n_xi]
            rho_j = self.rho[1, :, (receiver_idx -1) * self.n_xi: (receiver_idx) * self.n_xi]
            
            return Message(self.sender_id,  x_i=None, x_j=None, rho_i=rho_i, rho_j=rho_j, update='D')
        
        

class MessageReceiver:
    """
        Handle the communication phase between agents. Agents initialize this class and use its methods to receive messages from their neighbours.
        NOTE! the receipt and the process of the received message are computed in two different methods
    """
    
    #! This assumes that all the x_i have the same size
    
    def __init__(
            self, 
            receiver_id: int,
            adjacency_vector: np.ndarray,
            y_j: np.ndarray,
            rho_j: np.ndarray,
            n_xi: int
        ):

        self.receiver_id = receiver_id
        self.adjacency_vector = adjacency_vector
        self.y_j = y_j
        self.rho_j = rho_j        
        self.messages = []
        self.n_xi = n_xi
        
    def receive_message(self, message: Message):
        " Store the message received from neighbours in a local buffer"
        
        if message.sender_id == self.receiver_id:
            return
        
        self.messages.append(message)
        
    def process_messages(self, update: str)-> np.ndarray:
        " Reorder the messages received and return the local variables y and rho using the correct agent's ordering"        
                
        while not self.messages :
            message = self.messages.pop(0)
            receiver_idx = list(self.adjacency_vector).index(message.sender_id)
            if message.update == 'P' and update == 'P':
                self.y_j[0, :, receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi] = message.x_j
                self.y_j[1, :, receiver_idx * self.n_xi: (receiver_idx + 1) * self.n_xi] = message.x_i
                return self.y_j
            if message.update == 'D' and update == 'D':
                self.rho_j[0, :, (receiver_idx -1) * self.n_xi: (receiver_idx) * self.n_xi] = message.rho_j
                self.rho_j[1, :, (receiver_idx -1) * self.n_xi: (receiver_idx) * self.n_xi] = message.rho_i
                return self.rho_j
            if message.update == 'P' and update == 'D':
                raise ValueError("The update type must be the same")
            elif message.update == 'D' and update == 'P':
                raise ValueError("The update type must be the same")