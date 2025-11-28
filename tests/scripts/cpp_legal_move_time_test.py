from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.common import make_ndarray_board, make_int_board

from src.GUI import GUI, BoardGUI, init, chg_int_to_piece_gui
from src.GunjinShogi import Environment,JudgeBoard,TensorBoard, CppJudgeBoard

import torch
import numpy as np
import time

import GunjinShogiCore as GSC  
    
TEST_TIME = 10**6

def main():
    
    pieces_player1 = np.arange(PIECE_LIMIT, dtype=np.int32)
    pieces_player2 = np.arange(PIECE_LIMIT, dtype=np.int32)
    
    player1_tensor = make_ndarray_board(pieces_player1)
    player2_tensor = make_ndarray_board(pieces_player2)
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device=torch.device("cpu"))
    
    env = Environment(judge, tensorboard)
    env.set_board(player1_tensor, player2_tensor)
    
    all_time = 0
    for i in range(TEST_TIME):
        t1 = time.time()
        env.legal_move()  
        all_time += time.time() - t1
        
    print(f"試行回数:{TEST_TIME}回, 時間:{all_time}s")         
    
            
if __name__ == "__main__":
    main()