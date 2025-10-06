from src.common import LogData, Player

from abc import ABC, abstractmethod

from torch import Tensor

class IEnv(ABC):
    @abstractmethod
    def get_board_player1(self) -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player2(self) -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player_current(self) -> Tensor:
        pass
    
    @abstractmethod
    def legal_move(self) -> Tensor:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple[Tensor, LogData, bool]:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass
    
    @abstractmethod
    def set_board(self, board_player1: Tensor, board_player2: Tensor) -> None:
        pass
    
    @abstractmethod
    def get_current_player(self) -> Player:
        pass
    
    @abstractmethod
    def get_opponent_player(self) -> Player:
        pass