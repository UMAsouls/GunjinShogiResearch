from dataclasses import dataclass

import torch
import numpy as np

from src.const import BOARD_SHAPE
import GunjinShogiCore as GSC

import random
from collections import deque

@dataclass
class MiniBatch:
    boards: torch.Tensor
    rewards: torch.Tensor


class ReplayBuffer:
    def __init__(self, size:int = 10000, board_shape:tuple = [41,BOARD_SHAPE[0],BOARD_SHAPE[1]]):
        
        self.head = 0
        self.size = size
        self.length = 0

        self.boards = torch.zeros((size, board_shape), dtype=torch.float32)
        self.rewards = torch.zeros((size), dtype=torch.float32)
    
    def add(self, board: torch.Tensor, reward: float):
        self.boards[self.head] = board.cpu()
        self.rewards[self.head] = reward

        self.head = (self.head + 1) % self.size
        self.length = min(self.length+1, self.size)

    def sample(self, batch_size: int) -> MiniBatch:
        r = torch.randint(0, self.length, batch_size)

        batch = MiniBatch(self.boards[r], self.rewards[r])

        return batch
        
    def __len__(self):
        return self.length