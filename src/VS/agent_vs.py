from src.Interfaces import IAgent, IEnv

from src.common import make_tensor_board, Player

import numpy as np

def Agent_VS(agent1: IAgent, agent2: IAgent, env: IEnv, log_path:str = "") -> int:
    pieces1 = agent1.get_first_board()
    pieces2 = agent2.get_first_board()
    
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
        logs.append([log.action, log.player, log.erase, log.bef, log.aft])
        
    log_array = np.array(logs)
    
    if(log_path): np.save(log_path, log_array)
        
    if(env.get_current_player() == Player.PLAYER2): return 1
    elif(env.get_current_player() == Player.PLAYER1): return 2
    
    else: return 0
    
    