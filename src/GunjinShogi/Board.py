from src.GunjinShogi.Interfaces import IBoard

from src.common import EraseFrag, Player, get_action

from abc import abstractmethod

import numpy as np

import GunjinShogiCore as GSC

BUFFSIZE = 10000

class Board(IBoard):
    def __init__(self, size: tuple[int, int]):
        self._size = size
        
        self._s = self._size[0] * self._size[1]
        self._board_p1: np.ndarray = np.zeros(self._s, dtype=np.int32)
        self._board_p2: np.ndarray = np.zeros(self._s, dtype=np.int32)
        
        self._boards = (self._board_p1, self._board_p2)
        
        self._buffer: np.ndarray = np.zeros((BUFFSIZE, 5), dtype=np.int32)
        self._buf_idx: int = 0
        
    def erase(self, board: np.ndarray, pos:int) -> None:
        board[pos] = 0
    
    def move(self, board: np.ndarray, bef: int, aft: int) -> None:
        board[aft] = board[bef]
        board[bef] = 0
        
    def get_action(self, action:int) -> tuple[int, int]:
        return get_action(action)
    
    def get_opponent_action(self, bef:int, aft: int) -> tuple[int, int]:
        o_bef = self._s - bef - 1
        o_aft = self._s - aft - 1
        
        return o_bef,o_aft
    
    def get_plyaer_opponent_board(self, player: int) -> tuple[np.ndarray, np.ndarray]:
        oppose = 3 - player
        return self._boards[player-1], self._boards[oppose-1]
    
    def step(self, action: int, player: int, erase: EraseFrag):
        (bef, aft) = self.get_action(action)
        o_bef,o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        erased:int = -1
        if(erase == GSC.EraseFrag.BEF):
            self.erase(player_board, bef)
            self.erase(oppose_board, o_bef)
        elif(erase == GSC.EraseFrag.AFT):
            self.erase(player_board, aft)
            self.erase(oppose_board, o_aft)
        elif(erase == GSC.EraseFrag.BOTH):
            self.erase(player_board, bef)
            self.erase(oppose_board, o_bef)
            self.erase(player_board, aft)
            self.erase(oppose_board, o_aft)
        
        if(erase != GSC.EraseFrag.BEF):
            self.move(player_board, bef, aft)
            self.move(oppose_board, o_bef, o_aft)
        
        self._buffer[self._buf_idx] = np.array((action, player, int(erase), player_board[bef], oppose_board[aft]), dtype=np.int32)
        
        self._buf_idx += 1
        self._buf_idx %= BUFFSIZE
        
    def reset(self):
        self._board_p1: np.ndarray = np.zeros(self._s, dtype=np.int32)
        self._board_p2: np.ndarray = np.zeros(self._s, dtype=np.int32)
        
        self._boards = (self._board_p1, self._board_p2)
        
    def undo(self):
        (action, player, erase, bef_v, aft_v) = self._buffer[self._buf_idx]
        
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
            
    def set_board(self, board_player1:np.ndarray, board_player2: np.ndarray):
        self._board_p1 = board_player1.copy()
        
        self._board_p2 = board_player2.copy()
        
        self._boards = (self._board_p1, self._board_p2)
            