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
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple[Tensor, bool]:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass