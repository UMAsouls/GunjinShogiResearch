from src.GunjinShogi.Interfaces import IJudgeBoard
from src.GunjinShogi.const import JUDGE_TABLE
from src.GunjinShogi.Board import Board

from src.const import Piece,PIECE_KINDS, GOAL_POS, ENTRY_HEIGHT, ENTRY_POS, BOARD_SHAPE, BOARD_SHAPE_INT
from src.common import EraseFrag, Player, get_action, make_reflect_pos, get_opponent, change_pos_int_to_tuple, Config

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
    
    def step(self, action: int, player: GSC.Player, erase: GSC.EraseFrag) -> GSC.BattleEndFrag:
        return self.cppJudge.step(action, player, erase)
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        self.cppJudge.setBoard(board_player1, board_player2)
    
    def judge(self, action: int, player: GSC.Player) -> GSC.EraseFrag:
        
        judge = self.cppJudge.getJudge(action, player)
        erase: GSC.EraseFrag
        if(judge == GSC.JudgeFrag.PIECE_WIN): erase = GSC.EraseFrag.AFT
        elif(judge == GSC.JudgeFrag.PIECE_LOSE): erase = GSC.EraseFrag.BEF
        else: erase = GSC.EraseFrag.BOTH
        
        return erase
    
    def legal_move(self, player: GSC.Player) -> np.ndarray:
        legals = self.cppJudge.getLegalMove(player)
        return legals
    
    def get_piece_effected_by_action(self, action:int, player:GSC.Player) -> tuple[int,int]:
        f,t = get_action(action)
        width = Config.board_shape[0]
        fx,fy = (f%width, f//width)
        tx,ty = (t%width, t//width)
        
        r_tx,r_ty = make_reflect_pos((tx,ty))
        
        o_player = GSC.Player.PLAYER_ONE if player == GSC.Player.PLAYER_TWO else GSC.Player.PLAYER_TWO
        
        return self.cppJudge.get(fx,fy,player), self.cppJudge.get(r_tx, r_ty, o_player)
    
    def is_win(self, player:GSC.Player) -> GSC.BattleEndFrag:
        return self.cppJudge.isWin(player)
    
    def get_defined_board(self, pieces: np.ndarray, player: GSC.Player) -> "CppJudgeBoard":
        ncppJudge = self.cppJudge.getDefinedBoard(pieces, player)
        return CppJudgeBoard(ncppJudge)
    
    def get_int_board(self, p:GSC.Player) -> np.ndarray:
        return self.cppJudge.get_int_board(p)