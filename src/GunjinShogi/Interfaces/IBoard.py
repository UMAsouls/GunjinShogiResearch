from abc import ABC, abstractmethod

from torch import Tensor

class IBoard(ABC):
    @abstractmethod
    def reset() -> None:
        pass
    
    @abstractmethod
    def step(action: int) -> tuple[Tensor, bool]:
        pass
    
    @abstractmethod
    def undo() -> bool:
        pass