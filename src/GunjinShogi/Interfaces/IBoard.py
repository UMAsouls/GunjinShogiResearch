from src.common import EraseFrag

from abc import ABC, abstractmethod

from torch import Tensor

class IBoard(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: int, player: int, erase: EraseFrag):
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass
    
    @abstractmethod
    def set_board(self, board_player1: Tensor, board_player2: Tensor) -> None:
        pass