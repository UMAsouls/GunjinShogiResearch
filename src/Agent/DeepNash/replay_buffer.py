
from dataclasses import dataclass

import torch
import numpy as np

import GunjinShogiCore as GSC

@dataclass
class Trajectory:
    board: torch.Tensor
    action: int
    reward: float
    policy: torch.Tensor
    player: GSC.Player
    non_legal: torch.Tensor
    
class Episode:
    def __init__(self):
        self.boards : torch.Tensor
        self.actions: np.typing.NDArray[np.int32]
        self.rewards: np.typing.NDArray[np.int32]
        self.policies: np.typing.NDArray[np.int32]
        self.non_legals: torch.Tensor

class ReplayBuffer:
    def __init__(self, size:int = 100000):
        pass