from src.Interfaces import IEnv
from src.common import LogData, Player
from src.const import PIECE_DICT

from src.GunjinShogi.Interfaces import IJudgeBoard, ITensorBoard

import torch

import numpy as np

import GunjinShogiCore as GSC

MAX_STEP = 2000

def get_int_player(p:GSC.Player) -> int:
    return 1 if p == GSC.Player.PLAYER_ONE else 2

class Environment(IEnv):
    def __init__(self, judge_board: IJudgeBoard, tensor_board: ITensorBoard, deploy = True):
        self.judge_board = judge_board
        self.tensor_board = tensor_board
        
        self.player = GSC.Player.PLAYER_ONE
        self.winner = None
        
        self.steps:int = 0
        
        self.deploy = deploy
        
    def _player_change(self) -> None:
        self.player = self.get_opponent_player()
        
    def get_board_player1(self) -> np.ndarray:
        return self.tensor_board.get_board_player1()
    
    def get_board_player2(self) -> np.ndarray:
        return self.tensor_board.get_board_player2()
    
    def get_board_player_current(self) -> np.ndarray:
        pass
    
    def legal_move(self) -> np.ndarray:
        return self.judge_board.legal_move(get_int_player(self.player))
    
    def reset(self) -> None:
        self.judge_board.reset()
        self.tensor_board.reset()
        self.steps = 0
        self.player = GSC.Player.PLAYER_ONE
        self.winner = None
        self.deploy = True
         
    def deploy_step(self, action: int) -> tuple[np.ndarray, LogData, GSC.BattleEndFrag]:
        erase = self.judge_board.judge(action, get_int_player(self.player))
        
        frag = self.judge_board.step(action, get_int_player(self.player), erase)
        
        self.tensor_board.deploy_set(PIECE_DICT[action], self.player)
        
        self.steps += 1
        
        if(frag == GSC.BattleEndFrag.DEPLOY_END):
            self.deploy = False
            
        log = LogData(action, get_int_player(self.player), erase, 0, 0)
        
        self._player_change()
        
        tensor = self.get_tensor_board_current()
        
        return (tensor, log, frag)
    
    
    def step(self, action: int) -> tuple[np.ndarray, LogData, GSC.BattleEndFrag]:
        if(self.deploy):
            return self.deploy_step(action)
        
        if(action == -1):
            log = LogData(action, self.player, GSC.EraseFrag.BOTH, 0, 0)
            done = GSC.BattleEndFrag.LOSE
            self.winner = self.get_opponent_player()
            return (self.get_board_player_current(), log, done)
        
        erase = self.judge_board.judge(action, get_int_player(self.player))
        
        bef_piece, aft_piece = self.judge_board.get_piece_effected_by_action(action, get_int_player(self.player))
        
        self.judge_board.step(action, get_int_player(self.player), erase)
        self.tensor_board.step(action, get_int_player(self.player), erase)
        
        tensor = self.get_board_player_current()
        
        log = LogData(action, get_int_player(self.player), erase, bef_piece, aft_piece)
        
        done = self.judge_board.is_win(self.player)
        
        if(done == GSC.BattleEndFrag.WIN): self.winner = self.get_current_player()
        elif(done == GSC.BattleEndFrag.LOSE): self.winner = self.get_opponent_player()
        
        self.steps += 1
        
        if(self.steps == MAX_STEP):
            done = GSC.BattleEndFrag.DRAW
            self.winner = -1
        
        self._player_change()
        
        return (tensor, log, done)
    
    def undo(self) -> bool:
        self.judge_board.undo()
        self.tensor_board.undo()
        
        self._player_change()
        
    def set_board(self, board_player1, board_player2):
        self.judge_board.set_board(board_player1, board_player2)
        self.tensor_board.set_board(board_player1, board_player2)
        self.deploy = False
        
    def get_current_player(self):
        return self.player
    
    def get_opponent_player(self):
        return GSC.Player.PLAYER_ONE if self.player == GSC.Player.PLAYER_TWO else GSC.Player.PLAYER_TWO
    
    def get_winner(self):
        return self.winner
    
    def get_defined_env(self, pieces:np.ndarray, player:GSC.Player):
        defined = self.judge_board.get_defined_board(pieces, player)
        tensor_boad = self.tensor_board.get_defined_board(pieces, player, self.deploy)
        new_env = Environment(defined, tensor_boad, self.deploy)
        new_env.player = self.player
        return new_env
    
    def get_int_board(self) -> np.ndarray:
        return self.judge_board.get_int_board(self.player)
    
    def get_tensor_board_current(self):
        return self.tensor_board.get_board_player1() if self.player == GSC.Player.PLAYER_ONE else self.tensor_board.get_board_player2()