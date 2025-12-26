from abc import ABC, abstractmethod
from torch import Tensor

from src.GunjinShogi.Interfaces.IBoard import IBoard

import GunjinShogiCore as GSC

import numpy as np

class ITensorBoard(IBoard):
    
    @abstractmethod
    def deploy_set(self, piece, player:GSC.Player) -> None:
        pass
    
    @abstractmethod
    def deploy_end(self) -> None:
        pass
    
    @abstractmethod
    def get_defined_board(self, pieces: np.ndarray, player: GSC.Player, deploy = False) -> "ITensorBoard":
        pass
    
    @abstractmethod
    def set_max_step(self, max_step: int, max_non_attack: int):
        pass
    
    @abstractmethod
    def get_board(self, player:GSC.Player) -> Tensor:
        pass
    
    @abstractmethod
    def reset(self):
        pass