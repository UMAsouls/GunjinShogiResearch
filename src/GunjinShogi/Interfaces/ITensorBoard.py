from abc import ABC, abstractmethod
from torch import Tensor

from GunjinShogi.Interfaces.IBoard import IBoard

class ITensorBoard(IBoard):
    
    @abstractmethod
    def get_board_player1() -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player2() -> Tensor:
        pass