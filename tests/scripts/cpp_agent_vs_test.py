
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import LogMaker, make_ndarray_board

from src.Agent import RandomAgent,ISMCTSAgent
from src.VS import Cpp_Agent_VS

from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
import GunjinShogiCore as GSC

import numpy as np
import torch

BATTLES = 10

LOG_NAME = "cpp_random_test_1"

MODEL_DIR = "models/is_mcts"
MODEL_NANE = "v2/model_10000.pth"

HISTORY = 23

IN_CHANNELS = 18 + HISTORY
MID_CHANNELS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judgeboard = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, DEVICE, HISTORY)
    
    env = Environment(judgeboard, tensorboard)
    
    agent1 = ISMCTSAgent(GSC.Player.PLAYER_ONE, 0.7, 100,tensorboard.total_channels, MID_CHANNELS, f"{MODEL_DIR}/{MODEL_NANE}", DEVICE)
    #agent1 = RandomAgent()
    agent2 = RandomAgent()

    log_maker = LogMaker(LOG_NAME)

    wins1 = 0
    wins2 = 0
    
    for i in range(BATTLES):
        env.reset()
        
        pieces1 = agent1.get_first_board()
        pieces2 = agent2.get_first_board()
        board1 = make_ndarray_board(pieces1)
        board2 = make_ndarray_board(pieces2)
        
        log_maker.add_pieces(pieces1,pieces2)
    
        #env.set_board(board1, board2)
        
        a = env.judge_board.get_int_board(GSC.Player.PLAYER_ONE)

        win = Cpp_Agent_VS(agent1, agent2, env, log_maker)
        if(win == 1): wins1 += 1
        elif(win == 2): wins2 += 1
        
        print(f"BattleEnds: {i}/{BATTLES}")
        print(f"agent1: {wins1}回, agent2: {wins2}回")
        print(f"step数：{env.steps}")
        

    
if __name__ == "__main__":
    main()