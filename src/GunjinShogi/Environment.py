from src.Interfaces import IEnv
from src.common import LogData, Player

from src.GunjinShogi.Interfaces import IJudgeBoard, ITensorBoard

import torch

import numpy as np

import GunjinShogiCore as GSC

class Environment(IEnv):
    def __init__(self, judge_board: IJudgeBoard, tensor_board: ITensorBoard):
        self.judge_board = judge_board
        self.tensor_board = tensor_board
        
        self.player = Player.PLAYER1
        self.winner = None
        
    def _player_change(self) -> None:
        self.player = 3 - self.player
        
    def get_board_player1(self) -> np.ndarray:
        return self.tensor_board.get_board_player1()
    
    def get_board_player2(self) -> np.ndarray:
        return self.tensor_board.get_board_player2()
    
    def get_board_player_current(self) -> np.ndarray:
        pass
    
    def legal_move(self) -> np.ndarray:
        return self.judge_board.legal_move(int(self.player))
    
    def reset(self) -> None:
        self.judge_board.reset()
        self.tensor_board.reset()
    
    def step(self, action: int) -> tuple[np.ndarray, LogData, GSC.BattleEndFrag]:
        erase = self.judge_board.judge(action, int(self.player))
        
        bef_piece, aft_piece = self.judge_board.get_piece_effected_by_action(action, int(self.player))
        
        self.judge_board.step(action, int(self.player), erase)
        self.tensor_board.step(action, int(self.player), erase)
        
        tensor = self.get_board_player_current()
        
        log = LogData(action, self.player, erase, bef_piece, aft_piece)
        
        done = self.judge_board.is_win(self.player)
        
        if(done == GSC.BattleEndFrag.WIN): self.winner = self.get_current_player()
        elif(done == GSC.BattleEndFrag.LOSE): self.winner = self.get_opponent_player()
        
        self._player_change()
        
        return (tensor, log, done)
    
    def undo(self) -> bool:
        self.judge_board.undo()
        self.tensor_board.undo()
        
        self._player_change()
        
    def set_board(self, board_player1, board_player2):
        self.judge_board.set_board(board_player1, board_player2)
        self.tensor_board.set_board(board_player1, board_player2)
        
    def get_current_player(self):
        return self.player
    
    def get_opponent_player(self):
        return Player.PLAYER1 if self.player == Player.PLAYER2 else Player.PLAYER2
    
    def get_winner(self):
        return self.winner