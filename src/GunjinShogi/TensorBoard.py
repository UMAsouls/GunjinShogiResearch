from GunjinShogi.Interfaces import ITensorBoard

from GunjinShogi.Board import Board

import torch

class TensorBoard(Board, ITensorBoard):
    def __init__(self, size: tuple[int, int], device: torch.device):
        super().__init__(size, device)
        
    def get_board_player1(self) -> torch.Tensor:
        return self._board_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._board_p2
    
        
        
        
        
    
    