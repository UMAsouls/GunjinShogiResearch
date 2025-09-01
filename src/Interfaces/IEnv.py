from abc import ABC, abstractmethod

from torch import Tensor

class IEnv(ABC):
    @abstractmethod
    def get_board_player1() -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player2() -> Tensor:
        pass
    
    @abstractmethod
    def get_board_player_current() -> Tensor:
        pass
    
    @abstractmethod
    def reset() -> None:
        pass
    
    @abstractmethod
    def step(action: int) -> tuple[Tensor, bool]:
        pass
    
    @abstractmethod
    def undo() -> bool:
        pass