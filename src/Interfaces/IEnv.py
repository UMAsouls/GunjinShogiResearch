from src.common import LogData, Player

from abc import ABC, abstractmethod

from torch import Tensor

import numpy as np

import GunjinShogiCore as GSC

class IEnv(ABC):
    @abstractmethod
    def get_board_player1(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_board_player2(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_board_player_current(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def legal_move(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, LogData, GSC.BattleEndFrag]:
        pass
    
    @abstractmethod
    def undo(self) -> bool:
        pass
    
    @abstractmethod
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def get_current_player(self) -> GSC.Player:
        pass
    
    @abstractmethod
    def get_opponent_player(self) -> GSC.Player:
        pass
    
    @abstractmethod
    def get_winner(self) -> GSC.Player:
        pass
    
    @abstractmethod
    def get_defined_env(self, pieces:np.ndarray, player:GSC.Player) -> "IEnv":
        pass
    
    @abstractmethod
    def get_int_board(self, player: GSC.Player) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_tensor_board_current(self) -> Tensor:
        pass
    
    @abstractmethod
    def is_deploy(self) -> bool:
        pass