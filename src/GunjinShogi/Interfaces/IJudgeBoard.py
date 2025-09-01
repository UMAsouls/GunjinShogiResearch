from abc import ABC, abstractmethod
from torch import Tensor

from GunjinShogi.Interfaces.IBoard import IBoard



class IJudgeBoard(IBoard):
    
    @abstractmethod
    def judge(self, action: int, player: int) -> bool:
        pass
    
    @abstractmethod
    def legal_move(self, player: int) -> Tensor:
        pass