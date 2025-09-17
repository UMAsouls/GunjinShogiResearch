from src.const import BOARD_SHAPE,ENTRY_HEIGHT

from src.GUI.Piece import PieceGUI

def chg_int_to_piece_gui(int_board: list[list[int]]):
    board = [[None for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    for y,i in enumerate(int_board):
        for x,j in enumerate(i):
            if(j <= 0): continue
            
            dir = 0
            appear = True
            if(y < ENTRY_HEIGHT):
                dir = 180
                appear = False
            
            piece = PieceGUI(j,dir,appear)
            board[y][x] = piece
    
    return board