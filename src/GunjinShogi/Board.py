from GunjinShogi.Interfaces import IBoard
from GunjinShogi.const import EraseFrag

from abc import abstractmethod
import torch

import numpy as np

BUFFSIZE = 10000

class Board(IBoard):
    def __init__(self, size: tuple[int, int], device: torch.device):
        self._board_p1: torch.Tensor = torch.zeros(size, dtype=torch.int32, device=device)
        self._board_p2: torch.Tensor = torch.zeros(size, dtype=torch.int32, device=device)
        
        self._boards = (self._board_p1, self._board_p2)
        
        self._size = size
        self._device = device
        
        self._s = self._size[0] * self._size[1]
        
        self._buffer: torch.Tensor = torch.zeros((BUFFSIZE, 4), dtype=torch.int32)
        self._buf_idx: int = 0
        
    def erase(self, board: torch.Tensor, pos:int) -> None:
        board[pos] = 0
    
    def move(self, board: torch.Tensor, bef: int, aft: int) -> None:
        board[bef], board[aft] = board[aft], board[bef]
        
    def get_action(self, action:int) -> tuple[int, int]:
        (bef, aft) = [action//self._s, action%self._s]
        
        return bef,aft
    
    def step(self, action: int, player: int, erase: EraseFrag):
        (bef, aft) = self.get_action(action)
        
        o_bef = self._s - bef - 1
        o_aft = self._s - aft - 1
        
        oppose = 3 - player
        player_board = self._boards[player-1]
        oppose_board = self._boards[oppose-1]
        
        erased:int = -1
        
        if(erase == EraseFrag.BEFORE):
            erased = player_board[bef]
            self.erase(player_board, bef)
            self.erase(oppose_board, o_bef)
        elif(erase == EraseFrag.AFTER):
            erased = oppose_board[o_aft]
            self.erase(player_board, aft)
            self.erase(oppose_board, o_aft)
        
        self.move(player_board, bef, aft)
        self.move(oppose_board, o_bef, o_aft)
        
        self._buffer[self._buf_idx] = torch.as_tensor((action, player, int(erase), erased), dtype=torch.int32)
        
        self._buf_idx += 1
        self._buf_idx %= BUFFSIZE
        
    def reset(self):
        self._board_p1: torch.Tensor = torch.zeros(self._size, dtype=torch.int32, device=self._device)
        self._board_p2: torch.Tensor = torch.zeros(self._size, dtype=torch.int32, device=self._device)
        
        self._boards = (self._board_p1, self._board_p2)
        
    def undo(self):
        (action, player, erase, erased) = self._buffer[self._buf_idx].tolist()
        
        (bef, aft) = self.get_action(action)
        
        o_bef = self._s - bef - 1
        o_aft = self._s - aft - 1
        
        oppose = 3 - player
        player_board = self._boards[player-1]
        oppose_board = self._boards[oppose-1]
        
        self.move(player_board, aft, bef)
        self.move(oppose_board, o_aft, o_bef)
        
        if(erase == EraseFrag.BEFORE):
            player_board[bef] = erased
            oppose_board[o_bef] = -1
        elif(erase == EraseFrag.AFTER):
            player_board[aft] = -1
            oppose_board[o_aft] = erased