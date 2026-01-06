from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.GUI import PlayGUI, BoardGUI, init, chg_int_to_piece_gui, DeployGUI, AgentVsGUI
from src.GunjinShogi import Environment, CppJudgeBoard, JUDGE_TABLE

from src.Interfaces import IAgent
from src.Agent import RandomAgent, DeepNashAgent, DeepNashCnnAgent, ISMCTSAgent
from src.Agent.DeepNash import TensorBoard

from src.common import \
    make_int_board, make_ndarray_board,change_pos_tuple_to_int, make_reflect_pos, \
    LogMaker, LogData, Config

import torch

import numpy as np

import GunjinShogiCore as GSC

T_BOARD = TensorBoard
C_AGENT = DeepNashCnnAgent

MODEL_DIR = "models"

DEEPNASH_MODEL_NAME = "deepnash_mp/mini_cnn_t_v11/model_100000.pth"
ISMCTS_MODEL_NANE = "is_mcts/v2/model_100000.pth"

HISTORY = 20

IN_CHANNELS = T_BOARD.get_tensor_channels(HISTORY)
MID_CHANNELS = IN_CHANNELS*3//2

LOG_NAME = "human_vs_dp1"

CONFIG_PATH = "mini_board_config2.json"

PLAYER_FIRST = False

def make_int_board(board: np.ndarray) -> list[list[int]]:
    int_board = [[0 for i in range(Config.board_shape[0])] for j in range(Config.board_shape[1])]
    for y in range(Config.board_shape[1]):
        for x in range(Config.board_shape[0]):
            int_board[y][x] = board[change_pos_tuple_to_int(x,y)]
    return int_board


def deploy_phase(env: Environment, agent:IAgent, player_pieces: list[int], log_maker:LogMaker, player_turn = GSC.Player.PLAYER_ONE):
    idx = 0
    while env.deploy:
        if(env.get_current_player() == player_turn):
            action = player_pieces[idx]
            idx += 1
        else:
            action = agent.get_action(env)
        
        _, log, frag = env.step(action)
        log_maker.add_step(log)
        
def change_one_board(board1: np.ndarray, board2: np.ndarray):
    board = board1.copy()
    
    for y in range(Config.entry_height):
        for x in range(Config.board_shape[0]):
            rx,ry = make_reflect_pos((x,y))
            board[y][x] = board2[ry][rx]
            
    return board

def main():
    Config.load(CONFIG_PATH,JUDGE_TABLE)
    
    screen = init()
    screen_rect = screen.get_rect()

    pieces = np.arange(Config.piece_limit)
    
    board = make_int_board(make_ndarray_board(pieces))
    
    board_gui_int = chg_int_to_piece_gui(board, False)
    
    boardgui = BoardGUI(board_gui_int, screen_rect.center)
    deploy_gui = DeployGUI(boardgui, pieces)
    
    first_piece = deploy_gui.main_loop(screen)
    
    cppJudge = GSC.MakeJudgeBoard(CONFIG_PATH)
    judge = CppJudgeBoard(cppJudge)
    tensorboard = T_BOARD(Config.board_shape, device=torch.device("cpu"), history=HISTORY)
    
    env = Environment(judge)
    
    #agent = RandomAgent()
    #agent = ISMCTSAgent(GSC.Player.PLAYER_ONE, 0.7, 500,tensorboard.total_channels, MID_CHANNELS, f"{MODEL_DIR}/{ISMCTS_MODEL_NANE}")
    agent = C_AGENT(tensorboard.total_channels, MID_CHANNELS, torch.device("cpu"), tensorboard)
    agent.load_model(f"{MODEL_DIR}/{DEEPNASH_MODEL_NAME}")
    
    log_maker = LogMaker(LOG_NAME)
    
    player_turn = GSC.Player.PLAYER_ONE if PLAYER_FIRST else GSC.Player.PLAYER_TWO
    
    deploy_phase(env=env, agent=agent, player_pieces=first_piece, log_maker=log_maker, player_turn=player_turn)
    
    hide_player = GSC.Player.PLAYER_TWO if PLAYER_FIRST else GSC.Player.PLAYER_ONE
    
    int_board_1 = env.get_int_board(GSC.Player.PLAYER_ONE)
    int_board_2 = env.get_int_board(GSC.Player.PLAYER_TWO)
    piece_board = chg_int_to_piece_gui(change_one_board(int_board_1, int_board_2), True, hide_player=hide_player)
    
    boardgui = BoardGUI(piece_board, screen_rect.center)
    gui = AgentVsGUI(boardgui, env, agent, log_maker, PLAYER_FIRST)
    
    gui.main_loop(screen)
    
    log_maker.save()       
    
            
if __name__ == "__main__":
    main()