from abc import ABC, abstractmethod
from torch import Tensor

from src.GunjinShogi.Interfaces.IBoard import IBoard
from src.common import EraseFrag

import numpy as np

import GunjinShogiCore as GSC

class IJudgeBoard(IBoard):
    
    @abstractmethod
    def judge(self, action: int, player: int) -> EraseFrag:
        pass
    
    @abstractmethod
    def legal_move(self, player: int) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_piece_effected_by_action(self, action:int, player:int) -> tuple[int,int]:
        pass
    
    @abstractmethod
    def is_win(self, player:int) -> GSC.BattleEndFrag:
        pass
    
    @abstractmethod
    def set_state_from_IS(self, pieces: np.ndarray, player: int) -> None:
        pass
    
    @abstractmethod
    def is_state_from_IS(self) -> bool:
        pass
    
    @abstractmethod
    def turn_to_true_state(self) -> None:
        pass