from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.common import make_int_board, make_ndarray_board, LogMaker

from src.GUI import LogPlayGUI, BoardGUI, init, chg_int_to_piece_gui
from src.GunjinShogi import Environment,JudgeBoard,TensorBoard, CppJudgeBoard

import GunjinShogiCore as GSC

import torch

LOG_NAME = "random_test_1"

def main():
    screen = init()
    screen_rect = screen.get_rect()
    
    pieces1, pieces2, steps = LogMaker.load(LOG_NAME)
    
    int_board = make_int_board(pieces1, pieces2)
    
    piece_board = chg_int_to_piece_gui(int_board, False)
    
    player1_tensor = make_ndarray_board(pieces1)
    player2_tensor = make_ndarray_board(pieces2)
    
    cppJudge = GSC.MakeJudgeBoard(pieces1, pieces2, "config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device=torch.device("cpu"))
    
    env = Environment(judge, tensorboard)
    env.set_board(player1_tensor, player2_tensor)
    
    boardgui = BoardGUI(piece_board, screen_rect.center)
    gui = LogPlayGUI(boardgui, env, steps, 0.5)
    
    gui.main_loop(screen)            
    
            
if __name__ == "__main__":
    main()