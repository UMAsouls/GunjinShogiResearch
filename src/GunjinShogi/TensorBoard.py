from src.const import BOARD_SHAPE
from src.GunjinShogi.Interfaces import ITensorBoard

from src.GunjinShogi.Board import Board

from src.common import EraseFrag

import numpy as np
import torch

import GunjinShogiCore as GSC

class TensorBoard(ITensorBoard):
    def __init__(self, size: tuple[int, int], device: torch.device, en_history = 30):
        
        self._device = device

        self._board_p1 = torch.zeros([BOARD_SHAPE[0],BOARD_SHAPE[1],16 + 1 + en_history])
        self._board_p2 = torch.zeros([BOARD_SHAPE[0],BOARD_SHAPE[1],16 + 1 + en_history])

        self.first_p1: np.typing.NDArray[np.int32] = np.zeros(22)
        self.first_p2: np.typing.NDArray[np.int32] = np.zeros(22)
        
    def reset(self) -> None:
        pass
    
    def step(self, action: int, player: int, erase: GSC.EraseFrag):
        pass
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        pass
        
    def get_board_player1(self) -> torch.Tensor:
        return self._board_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._board_p2
    
        
        
        
        
    
    