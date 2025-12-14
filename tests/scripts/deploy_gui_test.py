from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.common import\
    make_int_board, make_ndarray_board, LogMaker, LogData, make_reflect_pos, \
    change_pos_tuple_to_int

from src.GUI import DeployGUI, BoardGUI, init, chg_int_to_piece_gui
from src.GunjinShogi import Environment,JudgeBoard,TensorBoard, CppJudgeBoard

import GunjinShogiCore as GSC

import torch
import numpy as np
import os

LOG_NAME = "cpp_random_test_1"

def make_int_board(board: np.ndarray) -> list[list[int]]:
    int_board = [[0 for i in range(BOARD_SHAPE[0])] for j in range(BOARD_SHAPE[1])]
    for y in range(BOARD_SHAPE[1]):
        for x in range(BOARD_SHAPE[0]):
            int_board[y][x] = board[change_pos_tuple_to_int(x,y)]
    return int_board



def main():
    print(f'PID: {os.getpid()}') 
    
    screen = init()
    screen_rect = screen.get_rect()

    pieces = np.arange(PIECE_LIMIT)
    
    board = make_int_board(make_ndarray_board(pieces))
    
    board_gui_int = chg_int_to_piece_gui(board, False)
    
    boardgui = BoardGUI(board_gui_int, screen_rect.center)
    deploy_gui = DeployGUI(boardgui, pieces)
    
    first_piece = deploy_gui.main_loop(screen)
    
    print(first_piece)
    
            
if __name__ == "__main__":
    main()