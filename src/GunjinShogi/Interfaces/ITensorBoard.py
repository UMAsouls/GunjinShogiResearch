from abc import ABC, abstractmethod
from torch import Tensor

from src.GunjinShogi.Interfaces.IBoard import IBoard

class ITensorBoard(IBoard):
    
    @abstractmethod
    def get_board_player1(self) -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player2(self) -> Tensor:
        pass