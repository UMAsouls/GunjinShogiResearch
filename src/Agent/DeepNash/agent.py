
from src.Interfaces import IAgent
from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer, Trajectory, Episode

from src.const import BOARD_SHAPE_INT

import torch
import torch.optim as optim

import numpy as np

class DeepNashAgent(IAgent):
    def __init__(
        self, in_channels: int, mid_channels: int, device: torch.device,
        lr: float = 0.01, 
    ):
        self.network = DeepNashNetwork(in_channels, mid_channels)
        self.optimizer = optim.Adam(self.network.parameters(), lr)
        
        self.device = device
        
    def learn(self):
        
        self.network.train()
        pass
        
    def v_trace(self, episode: Episode):
        pass
    
    def reward_transform(self, reward, policy):
        pass
        
    def get_action(self, env):
        legals = env.legal_move()
        
        non_legal = np.ones((BOARD_SHAPE_INT**2),dtype=np.bool_)
        non_legal[legals] = False
        
        non_legal_tensor = torch.from_numpy(non_legal).clone()
        non_legal_tensor = non_legal_tensor.to(self.device)
        
        board: torch.Tensor
        
        logit,value = self.network.forward(board, non_legal_tensor)
        policy = torch.softmax(logit)
        
    
    def get_first_board(self):
        return super().get_first_board()