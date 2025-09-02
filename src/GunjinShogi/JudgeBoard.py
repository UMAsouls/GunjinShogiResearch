from GunjinShogi.Interfaces import IJudgeBoard
from GunjinShogi.const import Piece, EraseFrag, JudgeFrag, JUDGE_TABLE
from GunjinShogi.Board import Board

import torch


class JudgeBoard(Board, IJudgeBoard):
    def __init__(self, size, device):
        device = torch.device("cpu")
        super().__init__(size, device)
        
        self._judge_table_p1 = JUDGE_TABLE
        self._judge_table_p2 = JUDGE_TABLE
        
        self._judge_tables = (self._judge_table_p1, self._judge_table_p2)
        
    def legal_move(self, player):
        return super().legal_move(player)
        
    def judge(self, action: int, player: int):
        bef,aft = self.get_action(action)
        o_bef, o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        player_table = self._judge_tables[player-1]
        
        p1 = player_board[bef]
        p2 = oppose_board[o_aft]
        if(p2 == int(Piece.Space)): return EraseFrag.NO 
        
        if(player_table[p1][p2] == JudgeFrag.Win): return EraseFrag.AFTER
        elif(player_table[p1][p2] == JudgeFrag.Lose): return EraseFrag.BEFORE
        else: return EraseFrag.BOTH
        
                
        