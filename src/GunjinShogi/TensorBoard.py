from src.GunjinShogi.Interfaces import ITensorBoard

from src.GunjinShogi.Board import Board

from src.common import EraseFrag

import numpy as np
import torch

class TensorBoard(ITensorBoard):
    def __init__(self, size: tuple[int, int], device: torch.device):
        
        self._device = device
        
    def reset(self) -> None:
        pass
    
    def step(self, action: int, player: int, erase: EraseFrag):
        pass
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        pass
        
    def get_board_player1(self) -> torch.Tensor:
        return self._board_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._board_p2
    
        
        
        
        
    
    