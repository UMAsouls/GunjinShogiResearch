
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
        sample_b = torch.zeros(board_tensor_shape, dtype=torch.float32)
        
        self.boards : torch.Tensor = sample_b.unsqueeze(0).expand(max_step,-1,-1,-1)
        self.actions: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)
        self.rewards: torch.Tensor = torch.zeros(max_step,dtype=torch.float32)
        self.policies: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.float32)
        self.non_legals: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.bool)

        self.t_effective:int = -1
        
        self.device = device
        self.head = 0
        
    def episode_end(self):
        self.boards = self.boards[:self.t_effective]
        self.actions = self.actions[:self.t_effective]
        self.rewards = self.rewards[:self.t_effective]
        self.policies = self.policies[:self.t_effective]
        self.non_legals = self.non_legals[:self.t_effective]
        
    def add_step(self, trac:Trajectory):
        #tensorはcpuに保存
        self.boards[self.head] = trac.board.cpu()
        self.actions[self.head] = trac.action
        self.rewards[self.head] = trac.reward
        self.policies[self.head] = trac.policy.cpu()
        self.non_legals[self.head] = trac.non_legal.cpu()
        
        self.t_effective += 1
        self.head += 1
        


class ReplayBuffer:
    def __init__(self, size:int = 10000):
        self.buffer = deque(maxlen=size)

    def add(self, episode: Episode):
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> list[Episode]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def __len__(self):
        return len(self.buffer)