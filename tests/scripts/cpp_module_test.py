from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.GUI import PlayGUI, BoardGUI, init, chg_int_to_piece_gui
from src.GunjinShogi import Environment,TensorBoard, CppJudgeBoard

from src.common import make_int_board, make_ndarray_board

import torch

import numpy as np

import GunjinShogiCore as GSC    

def main():
    screen = init()
    screen_rect = screen.get_rect()
    
    pieces_player1 = np.arange(PIECE_LIMIT, dtype=np.int32)
    pieces_player2 = np.arange(PIECE_LIMIT, dtype=np.int32)
    
    int_board = make_int_board(pieces_player1, pieces_player2)
    
    piece_board = chg_int_to_piece_gui(int_board)
    
    player1_tensor = make_ndarray_board(pieces_player1)
    player2_tensor = make_ndarray_board(pieces_player2)
    
    cppJudge = GSC.MakeJudgeBoard(pieces_player1, pieces_player2, "config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device=torch.device("cpu"))
    
    env = Environment(judge, tensorboard)
    env.set_board(player1_tensor, player2_tensor)
    
    boardgui = BoardGUI(piece_board, screen_rect.center)
    gui = PlayGUI(boardgui, env)
    
    gui.main_loop(screen)            
    
            
if __name__ == "__main__":
    main()