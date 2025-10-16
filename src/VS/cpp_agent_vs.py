from src.Interfaces import IAgent, IEnv
from src.common import make_ndarray_board, Player, LogMaker
from src.const import BOARD_SHAPE

from src.GunjinShogi import CppJudgeBoard, TensorBoard, Environment

import numpy as np
import torch

import GunjinShogiCore as GSC

def Cpp_Agent_VS(agent1: IAgent, agent2: IAgent, env:IEnv, log_maker:LogMaker) -> int:
    
    
    done = False
    agents = (agent1, agent2)
    logs = []
    while not done:
        player = agents[env.get_current_player() - 1]
        
        action = player.get_action(env)
        
        #actionが0以下 == 選択可能アクションが無い
        #このときは手番側の負け
        if(action < 0): 
            if(env.get_current_player() == Player.PLAYER1): return 1
            else: return 2
        
        _,log,frag = env.step(action)
        
        if(frag != GSC.BattleEndFrag.CONTINUE): done = True
        
        log_maker.add_step(log)
        
    log_maker.save()
        
    if(env.get_winner() == Player.PLAYER1): return 1
    elif(env.get_winner() == Player.PLAYER2): return 2
    
    else: return 0