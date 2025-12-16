from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

from src.GUI import PlayGUI, BoardGUI, init, chg_int_to_piece_gui, DeployGUI, AgentVsGUI
from src.GunjinShogi import Environment,TensorBoard, CppJudgeBoard

from src.Interfaces import IAgent
from src.Agent import RandomAgent, DeepNashAgent, ISMCTSAgent

from src.common import \
    make_int_board, make_ndarray_board,change_pos_tuple_to_int, make_reflect_pos, \
    LogMaker, LogData

import torch

import numpy as np

import GunjinShogiCore as GSC


MODEL_DIR = "models"

DEEPNASH_MODEL_NAME = "deepnash/v5/model_3500.pth"

HISTORY = 23

MID_CHANNELS = 40

LOG_NAME = "human_vs_random1"

def make_int_board(board: np.ndarray) -> list[list[int]]:
    int_board = [[0 for i in range(BOARD_SHAPE[0])] for j in range(BOARD_SHAPE[1])]
    for y in range(BOARD_SHAPE[1]):
        for x in range(BOARD_SHAPE[0]):
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
    
    for y in range(ENTRY_HEIGHT):
        for x in range(BOARD_SHAPE[0]):
            rx,ry = make_reflect_pos((x,y))
            board[y][x] = board2[ry][rx]
            
    return board

def main():
    
    screen = init()
    screen_rect = screen.get_rect()

    pieces = np.arange(PIECE_LIMIT)
    
    board = make_int_board(make_ndarray_board(pieces))
    
    board_gui_int = chg_int_to_piece_gui(board, False)
    
    boardgui = BoardGUI(board_gui_int, screen_rect.center)
    deploy_gui = DeployGUI(boardgui, pieces)
    
    first_piece = deploy_gui.main_loop(screen)
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device=torch.device("cpu"), history=HISTORY)
    
    env = Environment(judge, tensorboard)
    
    #agent = RandomAgent()
    agent = DeepNashAgent(tensorboard.total_channels, MID_CHANNELS, torch.device("cpu"))
    agent.load_model(f"{MODEL_DIR}/{DEEPNASH_MODEL_NAME}")
    
    log_maker = LogMaker(LOG_NAME)
    
    deploy_phase(env=env, agent=agent, player_pieces=first_piece, log_maker=log_maker)
    
    int_board_1 = env.get_int_board(GSC.Player.PLAYER_ONE)
    int_board_2 = env.get_int_board(GSC.Player.PLAYER_TWO)
    piece_board = chg_int_to_piece_gui(change_one_board(int_board_1, int_board_2), True)
    
    boardgui = BoardGUI(piece_board, screen_rect.center)
    gui = AgentVsGUI(boardgui, env, agent, log_maker, True)
    
    gui.main_loop(screen)
    
    log_maker.save()       
    
            
if __name__ == "__main__":
    main()