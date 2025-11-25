
from dataclasses import dataclass

import torch
import numpy as np

from src.const import BOARD_SHAPE_INT
import GunjinShogiCore as GSC

import random
from collections import deque

@dataclass
class Trajectory:
    board: torch.Tensor
    action: int
    reward: float
    policy: torch.Tensor
    player: GSC.Player
    non_legal: torch.Tensor
    
class Episode:
    def __init__(self, device: torch.device, board_tensor_shape:tuple, max_step = 2000):
        self.boards : torch.Tensor = torch.zeros((max_step, board_tensor_shape), dtype=torch.int16).to(device)
        self.actions: torch.Tensor = torch.zeros(max_step, dtype=torch.int32).to(device)
        self.rewards: torch.Tensor = torch.zeros(max_step,dtype=torch.float32).to(device)
        self.policies: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.float32).to(device)
        self.non_legals: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.bool).to(device)
        
        self.device = device
        self.head = 0
        
    def add_step(self, trac:Trajectory):
        self.boards[self.head] = trac.board
        self.actions[self.head] = trac.action
        self.rewards[self.head] = trac.reward
        self.policies[self.head] = trac.policy
        self.non_legals[self.head] = trac.non_legal
        
        self.head += 1
        


class ReplayBuffer:
    def __init__(self, size:int = 10000):
        self.buffer = deque(maxlen=size)

    def add(self, episode: Episode):
        self.buffer.append(episode)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def __len__(self):
        return len(self.buffer)