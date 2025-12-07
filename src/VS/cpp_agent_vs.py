from src.Interfaces import IAgent, IEnv
from src.common import make_ndarray_board, Player, LogMaker,get_action, change_pos_int_to_tuple
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
        player = agent1 if env.get_current_player() == GSC.Player.PLAYER_ONE else agent2
        
        action = player.get_action(env)
        
        #actionが0以下 == 選択可能アクションが無い
        #このときは手番側の負け
        if(action < 0): 
            if(env.get_current_player() == GSC.Player.PLAYER_ONE): return 1
            else: return 2
        
        _,log,frag = env.step(action)
        
        if(frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END): done = True
        
        log_maker.add_step(log)
        
    log_maker.save()
        
    if(env.get_winner() == GSC.Player.PLAYER_ONE): return 1
    elif(env.get_winner() == GSC.Player.PLAYER_TWO): return 2
    elif(env.get_winner() == -1): return -1
    
    else: return 0