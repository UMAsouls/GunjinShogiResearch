from src.GUI import init, BoardGUI, PieceGUI

from src.GUI.const import WINDOW_SIZE, MASS_SIZE, BOARD_SHAPE
from src.GUI.assets import BoardSurface

from src.const import Piece

import pygame as pg
from pygame.locals import *

def chg_piece_board(int_board):
    board = [[None for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    for y,i in enumerate(int_board):
        for x,j in enumerate(i):
            if(j <= 0): continue
            
            piece = PieceGUI(j)
            board[y][x] = piece
    
    return board

def test():
    screen = init()
    
    int_board = [[0 for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    int_board[7][1] = Piece.General
    int_board[3][2] = Piece.Plane
    int_board[5][2] = Piece.Plane
    int_board[6][3] = Piece.LieutenantColonel
    
    piece_board = chg_piece_board(int_board)
    
    boardgui = BoardGUI(piece_board)
    
    done = False
    while not done:
        pg.display.update()
        boardgui.draw(screen)
            
        for event in pg.event.get():
            if(event.type == QUIT):
                done = True
                pg.quit()
                return
            
if __name__ == "__main__":
    test()   