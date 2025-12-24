from src.common import Config

from src.GUI.Piece import PieceGUI

def chg_int_to_piece_gui(int_board: list[list[int]], hide:bool = True):
    board = [[None for _ in range(Config.board_shape[0])] for _ in range(Config.board_shape[1])]
    
    for y,i in enumerate(int_board):
        for x,j in enumerate(i):
            if(j <= 0): continue
            
            dir = 0
            appear = True
            if(y < Config.entry_height):
                dir = 180
                if(hide): appear = False
            
            piece = PieceGUI(j,dir,appear)
            board[y][x] = piece
    
    return board