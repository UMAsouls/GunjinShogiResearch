from src.GunjinShogi.Interfaces import IJudgeBoard
from src.GunjinShogi.const import JudgeFrag, JUDGE_TABLE
from src.GunjinShogi.Board import Board

from src.const import Piece,PIECE_KINDS, GOAL_POS, ENTRY_HEIGHT, ENTRY_POS, BOARD_SHAPE
from src.common import EraseFrag, Player

import GunjinShogiCore as GSC

import numpy as np



def get_player(player: int) -> GSC.Player:
    c_player: GSC.Player
    if(player == Player.PLAYER1): c_player = GSC.Player.PLAYER_ONE
    else: c_player = GSC.Player.PLAYER_TWO
    return c_player

class CppJudgeBoard(IJudgeBoard):
    def __init__(self, cppJudge: GSC.JudgeBoard):
        self.cppJudge = cppJudge
        
    def reset(self) -> None:
        self.cppJudge.reset()
    
    def step(self, action: int, player: int, erase: EraseFrag) -> GSC.BattleEndFrag:
        c_erase: GSC.EraseFrag
        if(erase == EraseFrag.AFTER): c_erase = GSC.EraseFrag.AFT
        elif(erase == EraseFrag.AFTER): c_erase = GSC.EraseFrag.BEF
        else: c_erase = GSC.EraseFrag.BOTH
        
        c_player: GSC.Player = get_player(player)
        
        return self.cppJudge.step(action, c_player, c_erase)
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        pass
    
    def judge(self, action: int, player: int) -> EraseFrag:
        c_player: GSC.Player = get_player(player)
        
        return self.cppJudge.getJudge(action, c_player)
    
    def legal_move(self, player: int) -> np.ndarray:
        p = get_player(player)
        legals = self.cppJudge.getLegalMove(p)
        return legals
    
    def get_piece_effect_by_action(self, action:int, player:int) -> tuple[int,int]:
        pass
    
    def is_win(self, player:int) -> GSC.BattleEndFrag:
        return self.cppJudge.isWin(get_player(player))