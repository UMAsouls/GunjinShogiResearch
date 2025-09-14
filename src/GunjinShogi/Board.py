from src.GunjinShogi.Interfaces import IBoard

from src.common import EraseFrag, Player, get_action

from abc import abstractmethod
import torch

import numpy as np

BUFFSIZE = 10000

class Board(IBoard):
    def __init__(self, size: tuple[int, int], device: torch.device):
        self._s = self._size[0] * self._size[1]
        self._board_p1: torch.Tensor = torch.zeros(self._s, dtype=torch.int32, device=device)
        self._board_p2: torch.Tensor = torch.zeros(self._s, dtype=torch.int32, device=device)
        
        self._boards = (self._board_p1, self._board_p2)
        
        self._size = size
        self._device = device
        
        self._buffer: torch.Tensor = torch.zeros((BUFFSIZE, 5), dtype=torch.int32)
        self._buf_idx: int = 0
        
    def erase(self, board: torch.Tensor, pos:int) -> None:
        board[pos] = 0
    
    def move(self, board: torch.Tensor, bef: int, aft: int) -> None:
        board[aft] = board[bef]
        board[bef] = 0
        
    def get_action(self, action:int) -> tuple[int, int]:
        return get_action(action)
    
    def get_opponent_action(self, bef:int, aft: int) -> tuple[int, int]:
        o_bef = self._s - bef - 1
        o_aft = self._s - aft - 1
        
        return o_bef,o_aft
    
    def get_plyaer_opponent_board(self, player: int) -> tuple[torch.Tensor, torch.Tensor]:
        oppose = 3 - player
        return self._boards[player-1], self._boards[oppose-1]
    
    def step(self, action: int, player: int, erase: EraseFrag):
        (bef, aft) = self.get_action(action)
        o_bef,o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        erased:int = -1
        if(erase == EraseFrag.BEFORE):
            self.erase(player_board, bef)
            self.erase(oppose_board, o_bef)
        elif(erase == EraseFrag.AFTER):
            self.erase(player_board, aft)
            self.erase(oppose_board, o_aft)
        elif(erase == EraseFrag.BOTH):
            self.erase(player_board, bef)
            self.erase(oppose_board, o_bef)
            self.erase(player_board, aft)
            self.erase(oppose_board, o_aft)
        
        if(erase != EraseFrag.BEFORE):
            self.move(player_board, bef, aft)
            self.move(oppose_board, o_bef, o_aft)
        
        self._buffer[self._buf_idx] = torch.as_tensor((action, player, int(erase), player_board[bef], oppose_board[aft]), dtype=torch.int32)
        
        self._buf_idx += 1
        self._buf_idx %= BUFFSIZE
        
    def reset(self):
        self._board_p1: torch.Tensor = torch.zeros(self._size, dtype=torch.int32, device=self._device)
        self._board_p2: torch.Tensor = torch.zeros(self._size, dtype=torch.int32, device=self._device)
        
        self._boards = (self._board_p1, self._board_p2)
        
    def undo(self):
        (action, player, erase, bef_v, aft_v) = self._buffer[self._buf_idx].tolist()
        
        (bef, aft) = self.get_action(action)
        o_bef,o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        if(erase != EraseFrag.BEFORE):
            self.move(player_board, aft, bef)
            self.move(oppose_board, o_aft, o_bef)
        
        if(erase == EraseFrag.BEFORE):
            player_board[bef] = bef_v
            oppose_board[o_bef] = -1
        elif(erase == EraseFrag.AFTER):
            player_board[aft] = -1
            oppose_board[o_aft] = aft_v
        elif(erase == EraseFrag.BOTH):
            player_board[bef] = bef_v
            player_board[aft] = -1
            oppose_board[o_bef] = -1
            oppose_board[o_aft] = aft_v
            
    def set_board(self, board_player1, board_player2):
        self._board_p1 = board_player1
        self._board_p1.to(device=self._device)
        
        self._board_p2 = board_player2
        self._board_p2.to(device=self._device)
            