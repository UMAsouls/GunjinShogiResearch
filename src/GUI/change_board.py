from src.common import Config

from src.GUI.Piece import PieceGUI

import GunjinShogiCore as GSC

def chg_int_to_piece_gui(int_board: list[list[int]], hide:bool = True, hide_player:GSC.Player = GSC.Player.PLAYER_TWO):
    board = [[None for _ in range(Config.board_shape[0])] for _ in range(Config.board_shape[1])]
    
    for y,i in enumerate(int_board):
        for x,j in enumerate(i):
            if(j <= 0): continue
            
            dir = 0
            appear = False
            if(y < Config.entry_height):
                dir = 180
                if(hide_player == GSC.Player.PLAYER_ONE): appear = True
            else:
                if(hide_player == GSC.Player.PLAYER_TWO): appear = True
            
            if(not hide):
                appear = True
            
            piece = PieceGUI(j,dir,appear)
            board[y][x] = piece
    
    return board