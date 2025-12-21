from abc import ABC, abstractmethod

from src.Interfaces.IEnv import IEnv

from src.common import LogData

import torch
import numpy as np

class IAgent(ABC):
    @abstractmethod
    def get_action(self, env: IEnv) -> int:
        pass
    
    @abstractmethod
    def get_first_board(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def step(self, log:LogData):
        pass