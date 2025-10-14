from src.GunjinShogi.Interfaces import IJudgeBoard
from src.GunjinShogi.const import JudgeFrag, JUDGE_TABLE
from src.GunjinShogi.Board import Board

from src.const import Piece,PIECE_KINDS, GOAL_POS, ENTRY_HEIGHT, ENTRY_POS, BOARD_SHAPE, BOARD_SHAPE_INT
from src.common import EraseFrag, Player, get_action, make_reflect_pos, get_opponent

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
        elif(erase == EraseFrag.BEFORE): c_erase = GSC.EraseFrag.BEF
        else: c_erase = GSC.EraseFrag.BOTH
        
        c_player: GSC.Player = get_player(player)
        
        return self.cppJudge.step(action, c_player, c_erase)
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        pass
    
    def judge(self, action: int, player: int) -> EraseFrag:
        c_player: GSC.Player = get_player(player)
        
        judge = self.cppJudge.getJudge(action, c_player)
        erase: EraseFrag
        if(judge == GSC.JudgeFrag.PIECE_WIN): erase = EraseFrag.AFTER
        elif(judge == GSC.JudgeFrag.PIECE_LOSE): erase = EraseFrag.BEFORE
        else: erase = EraseFrag.BOTH
        
        return erase
    
    def legal_move(self, player: int) -> np.ndarray:
        p = get_player(player)
        legals = self.cppJudge.getLegalMove(p)
        return legals
    
    def get_piece_effected_by_action(self, action:int, player:int) -> tuple[int,int]:
        f,t = get_action(action)
        width = BOARD_SHAPE[0]
        fx,fy = (f%width, f//width)
        tx,ty = (t%width, t//width)
        
        r_tx,r_ty = make_reflect_pos((tx,ty))
        
        o_player = get_opponent(player)
        c_player = get_player(player)
        c_o_player = get_player(o_player)
        
        return self.cppJudge.get(fx,fy,c_player), self.cppJudge.get(r_tx, r_ty, c_o_player)
    
    def is_win(self, player:int) -> GSC.BattleEndFrag:
        return self.cppJudge.isWin(get_player(player))
    
    def set_state_from_IS(self, pieces: np.ndarray, player: int) -> None:
        pass
    
    def is_state_from_IS(self) -> bool:
        pass
    
    def turn_to_true_state(self) -> None:
        pass