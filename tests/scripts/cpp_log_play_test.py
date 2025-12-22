from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.common import make_int_board, make_ndarray_board, LogMaker, LogData, make_reflect_pos

from src.GUI import LogPlayGUI, BoardGUI, init, chg_int_to_piece_gui
from src.GunjinShogi import Environment,JudgeBoard,TensorBoard, CppJudgeBoard

import GunjinShogiCore as GSC

import torch
import numpy as np
import os

LOG_NAME = "cpp_deepnash_test_1"

def deploy_phase(env: Environment, steps: np.ndarray) -> int:
    idx = 0
    while env.deploy:
        data = steps[idx]
        log = LogData(data[0], data[1], data[2], data[3], data[4])
        
        env.step(log.action)
        
        idx += 1
        
    return idx


def change_one_board(board1: np.ndarray, board2: np.ndarray):
    board = board1.copy()
    
    for y in range(ENTRY_HEIGHT):
        for x in range(BOARD_SHAPE[0]):
            rx,ry = make_reflect_pos((x,y))
            board[y][x] = board2[ry][rx]
            
    return board


def main():
    print(f'PID: {os.getpid()}') 
    
    screen = init()
    screen_rect = screen.get_rect()
    
    pieces1, pieces2, steps = LogMaker.load(LOG_NAME)
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device=torch.device("cpu"))
    
    env = Environment(judge, tensorboard)
    
    idx = deploy_phase(env, steps)
    steps = steps[idx:]
    
    int_board_1 = env.get_int_board(GSC.Player.PLAYER_ONE)
    int_board_2 = env.get_int_board(GSC.Player.PLAYER_TWO)
    piece_board = chg_int_to_piece_gui(change_one_board(int_board_1, int_board_2), False)
    
    boardgui = BoardGUI(piece_board, screen_rect.center)
    gui = LogPlayGUI(boardgui, env, steps, 0.5)
    
    gui.main_loop(screen)            
    
            
if __name__ == "__main__":
    main()