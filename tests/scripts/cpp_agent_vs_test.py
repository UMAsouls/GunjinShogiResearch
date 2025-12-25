
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import LogMaker, make_ndarray_board, Config

from src.Agent import RandomAgent,ISMCTSAgent, DeepNashAgent, RuleBaseAgent
from src.VS import Cpp_Agent_VS

from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard, JUDGE_TABLE
import GunjinShogiCore as GSC

import numpy as np
import torch

BATTLES = 100

CONFIG_PATH = "mini_board_config.json"

Config.load(CONFIG_PATH,JUDGE_TABLE)

LOG_NAME = "cpp_mini_random_test_1"

MODEL_DIR = "models"
ISMCTS_MODEL_NANE = "is_mcts/v2/model_100000.pth"

DEEPNASH_MODEL_NAME = "deepnash_mp/mini_v8/model_100.pth"
DEEPNASH_MODEL_NAME2 = "deepnash_mp/v7/model_1815.pth"

HISTORY = 20

IN_CHANNELS = TensorBoard.get_tensor_channels(HISTORY)
MID_CHANNELS = IN_CHANNELS*2//3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    Config.load(CONFIG_PATH,JUDGE_TABLE)

    cppJudge = GSC.MakeJudgeBoard(CONFIG_PATH)
    judgeboard = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(Config.board_shape, torch.device("cpu"), HISTORY)
    
    env = Environment(judgeboard, tensorboard)
    
    agent1 = DeepNashAgent(tensorboard.total_channels, MID_CHANNELS, torch.device("cpu"))
    agent1.load_model(f"{MODEL_DIR}/{DEEPNASH_MODEL_NAME}")
    #agent1 = RandomAgent()
    #agent1 = ISMCTSAgent(GSC.Player.PLAYER_ONE, 0.7, 100,tensorboard.total_channels, MID_CHANNELS, f"{MODEL_DIR}/{MODEL_NANE}", DEVICE)
    #agent2 = RuleBaseAgent()
    agent2 = RandomAgent()
    #agent2 = DeepNashAgent(tensorboard.total_channels, MID_CHANNELS, torch.device("cpu"))
    #agent2.load_model(f"{MODEL_DIR}/{DEEPNASH_MODEL_NAME2}")

    wins1 = 0
    wins2 = 0
    
    for i in range(BATTLES):
        log_maker = LogMaker(LOG_NAME)
        
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