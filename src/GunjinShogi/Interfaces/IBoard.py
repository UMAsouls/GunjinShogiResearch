from abc import ABC, abstractmethod

from torch import Tensor

class IBoard(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: int, player: int) -> None:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass