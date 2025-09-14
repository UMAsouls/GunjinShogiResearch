from abc import ABC, abstractmethod
from torch import Tensor

from src.GunjinShogi.Interfaces.IBoard import IBoard
from src.common import EraseFrag



class IJudgeBoard(IBoard):
    
    @abstractmethod
    def judge(self, action: int, player: int) -> EraseFrag:
        pass
    
    @abstractmethod
    def legal_move(self, player: int) -> Tensor:
        pass
    
    @abstractmethod
    def get_piece_effect_by_action(self, action:int, player:int) -> tuple[int,int]:
        pass
    
    @abstractmethod
    def is_win(self, player:int) -> bool:
        pass