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

def emphasize_mouse_pos(gui:BoardGUI, mouse_pos: tuple[int,int]):
    onboard = gui.get_selected_pos(mouse_pos)
    gui.set_emp_pos(onboard)

def emphasize_clicked_pos(gui:BoardGUI, mouse_pos: tuple[int,int], legal_pos:list[tuple[int,int]]):
    onboard = gui.get_selected_pos(mouse_pos)
    
    gui.set_selected_pos(onboard)
    gui.set_legal_pos(legal_pos[onboard[1]][onboard[0]])

def test():
    screen = init()
    
    int_board = [[0 for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    int_board[7][1] = Piece.General
    int_board[3][2] = Piece.Plane
    int_board[5][2] = Piece.Plane
    int_board[6][3] = Piece.LieutenantColonel
    
    piece_board = chg_piece_board(int_board)
    
    legal_pos = [[[] for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    legal_pos[7][1].append((2,2))
    legal_pos[7][1].append((0,5))
    legal_pos[3][2].append((3,5))
    legal_pos[5][2].append((6,3))
    
    boardgui = BoardGUI(piece_board)
    
    done = False
    while not done:
        pg.display.update()
            
        for event in pg.event.get():
            if(event.type == QUIT):
                done = True
                pg.quit()
                return
        
        clicked = pg.mouse.get_pressed()[0]
        emphasize_mouse_pos(boardgui, pg.mouse.get_pos())
        if(clicked): emphasize_clicked_pos(boardgui, pg.mouse.get_pos(), legal_pos)
            
        boardgui.draw(screen)
            
if __name__ == "__main__":
    test()   