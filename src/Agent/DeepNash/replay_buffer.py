
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
    def __init__(self, board_tensor_shape:tuple, max_step = 2000):
        sample_b = torch.zeros(board_tensor_shape, dtype=torch.float32)
        
        self.boards : torch.Tensor = sample_b.unsqueeze(0).repeat(max_step,1,1,1)
        self.actions: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)
        self.rewards: torch.Tensor = torch.zeros((max_step,2),dtype=torch.float32)
        self.policies: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.float32)
        self.non_legals: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.bool)
        self.players: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)

        self.t_effective:int = -1
        self.head = 0
        
    def episode_end(self):
        self.boards = self.boards[:self.head]
        self.actions = self.actions[:self.head]
        self.rewards = self.rewards[:self.head]
        self.policies = self.policies[:self.head]
        self.non_legals = self.non_legals[:self.head]
        
    def add_step(self, trac:Trajectory):
        #tensorはcpuに保存
        self.boards[self.head] = trac.board.cpu()
        self.actions[self.head] = trac.action
        self.rewards[self.head] = trac.reward
        self.policies[self.head] = trac.policy.cpu()
        self.non_legals[self.head] = trac.non_legal.cpu()
        self.players[self.head] = 0 if trac.player == GSC.Player.PLAYER_ONE else 1
        
        self.t_effective += 1
        self.head += 1
        
    def set_reward(self, reward:float):
        r1 = reward
        r2 = 1-reward
        
        self.rewards[:,0] = r1
        self.rewards[:,1] = r2


@dataclass
class MiniBatch:
    boards: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    policies: torch.Tensor
    non_legals: torch.Tensor
    players: torch.Tensor


class ReplayBuffer:
    def __init__(self, size:int = 10000, max_step:int = 1000, board_shape:tuple = [41,6,9]):
        self.buffer = deque(maxlen=size)
        
        self.head = 0
        self.size = size
        self.max_step = max_step
        self.length = 0

    def add(self, episode: Episode):
        self.buffer.append(episode)
        
        self.head = (self.head + 1) % self.size
        self.length = min(self.length + 1, self.size)
        

    def sample(self, batch_size: int) -> list[Episode]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
        
    def __len__(self):
        return self.length
    
