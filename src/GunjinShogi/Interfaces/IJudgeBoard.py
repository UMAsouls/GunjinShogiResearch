from abc import ABC, abstractmethod
from torch import Tensor

from GunjinShogi.Interfaces.IBoard import IBoard
from GunjinShogi.const import EraseFrag



class IJudgeBoard(IBoard):
    
    @abstractmethod
    def judge(self, action: int, player: int) -> EraseFrag:
        pass
    
    @abstractmethod
    def legal_move(self, player: int) -> Tensor:
        pass