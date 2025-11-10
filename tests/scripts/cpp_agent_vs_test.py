
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import LogMaker, make_ndarray_board

from src.Agent import RandomAgent
from src.VS import Cpp_Agent_VS

from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
import GunjinShogiCore as GSC

import numpy as np
import torch

BATTLES = 100

LOG_NAME = "cpp_random_test_1"

def main():
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    log_maker = LogMaker(LOG_NAME)
    
    pieces1 = agent1.get_first_board()
    pieces2 = agent2.get_first_board()
    
    log_maker.add_pieces(pieces1, pieces2)

    cppJudge = GSC.MakeJudgeBoard(pieces1, pieces2, "config.json")
    judgeboard = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, torch.device("cpu"))
    
    env = Environment(judgeboard, tensorboard)
    
    board1 = make_ndarray_board(pieces1)
    board2 = make_ndarray_board(pieces2)
    
    env.set_board(board1, board2)

    wins1 = 0
    wins2 = 0
    
    for i in range(BATTLES):
        env.reset()
        win = Cpp_Agent_VS(agent1, agent2, env, log_maker)
        if(win == 1): wins1 += 1
        elif(win == 2): wins2 += 1
        
        print(f"BattleEnds: {i}/{BATTLES}")
        print(f"agent1: {wins1}回, agent2: {wins2}回")

    
if __name__ == "__main__":
    main()