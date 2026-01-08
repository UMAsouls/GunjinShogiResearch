
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import LogMaker, make_ndarray_board, Config

from src.Agent import RandomAgent,ISMCTSAgent, DeepNashAgent, RuleBaseAgent, SimpleRuleBaseAgent, DeepNashCnnAgent
from src.Agent.DeepNash import TensorBoard, SimpleTensorBoard
from src.VS import Cpp_Agent_VS

from src.GunjinShogi import Environment, CppJudgeBoard, JUDGE_TABLE
import GunjinShogiCore as GSC

import numpy as np
import torch
import os

T_BOARD = TensorBoard
T_BOARD2 = SimpleTensorBoard

BATTLES = 500

CONFIG_PATH = "mini_board_config2.json"

Config.load(CONFIG_PATH,JUDGE_TABLE)

LOG_NAME = "cpp_mini_random_test_1"

MODEL_DIR = "models"
ISMCTS_MODEL_NANE = "is_mcts/v2/model_100000.pth"

DEEPNASH_MODEL_NAME = "deepnash_mp/mini_cnn_t_v11/model_100000.pth"
DEEPNASH_MODEL_NAME2 = "deepnash_mp/mini_cnn_t_v11/model_100000.pth"

HISTORY = 20

IN_CHANNELS = T_BOARD.get_tensor_channels(HISTORY)
MID_CHANNELS = IN_CHANNELS*3//2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 400
NON_ATTACK_DRAW = 100

DATA_DIR = "data/rule_base_vs"
DATA_NAME = "ver1"

torch.set_printoptions(edgeitems=1000)

def main():
    Config.load(CONFIG_PATH,JUDGE_TABLE)

    cppJudge = GSC.MakeJudgeBoard(CONFIG_PATH)
    judgeboard = CppJudgeBoard(cppJudge)
    
    env = Environment(judgeboard, max_step=MAX_STEPS, max_non_attack=NON_ATTACK_DRAW)
    
    tensorboard = T_BOARD(Config.board_shape, torch.device("cpu"), HISTORY)
    tensorboard.set_max_step(MAX_STEPS,NON_ATTACK_DRAW)
    deepnash = DeepNashCnnAgent(tensorboard.total_channels, MID_CHANNELS, torch.device("cpu"), tensorboard)
    deepnash.load_model(f"{MODEL_DIR}/{DEEPNASH_MODEL_NAME}")
    simple = SimpleRuleBaseAgent()
    
    simple_first_pieces = [
        np.random.permutation(np.arange(Config.piece_limit)) for i in range(4)
    ]

    
    
    data = ""
    
    for pieces in simple_first_pieces:
        wins1 = 0
        wins2 = 0
        for i in range(BATTLES):
            log_maker = LogMaker(LOG_NAME)
        
            env.reset()
            deepnash.reset()
            simple.reset()
            
            simple.set_first_pieces(pieces)
        
            pieces1 = deepnash.get_first_board()
            pieces2 = simple.get_first_board()
        
            log_maker.add_pieces(pieces1,pieces2)
    
            #env.set_board(board1, board2)
        
            a = env.judge_board.get_int_board(GSC.Player.PLAYER_ONE)

            win = Cpp_Agent_VS(deepnash, simple, env, log_maker)
            if(win == 1): wins1 += 1
            elif(win == 2): wins2 += 1
        
        print(f"agent1: {wins1}回, agent2: {wins2}回")
        
        for p in pieces:
            data += f"{p} "
        data += f": {wins1}, {wins2}\n"
        
    path = f"{DATA_DIR}/{DATA_NAME}.txt"
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        f.write(data)
        f.close()   

    
if __name__ == "__main__":
    main()