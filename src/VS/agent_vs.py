from src.Interfaces import IAgent, IEnv

from src.common import make_tensor_board, Player, LogMaker

import numpy as np

def Agent_VS(agent1: IAgent, agent2: IAgent, env: IEnv, log_name:str = "") -> int:
    log_maker = LogMaker(log_name)
    
    pieces1 = agent1.get_first_board()
    pieces2 = agent2.get_first_board()
    
    
    log_maker.add_pieces(pieces1, pieces2)
    
    board1 = make_tensor_board(pieces1)
    board2 = make_tensor_board(pieces2)
    
    env.set_board(board1, board2)
    
    done = False
    agents = (agent1, agent2)
    logs = []
    while not done:
        player = agents[env.get_current_player() - 1]
        
        action = player.get_action(env)
        
        _,log,done = env.step(action)
        
        log_maker.add_step(log)
        
    if(log_name): log_maker.save()
        
    if(env.get_current_player() == Player.PLAYER2): return 1
    elif(env.get_current_player() == Player.PLAYER1): return 2
    
    else: return 0
    
    