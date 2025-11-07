from src.Interfaces import IAgent, IEnv
from src.const import PIECE_LIMIT

import numpy as np

class RandomAgent(IAgent):
    def __init__(self):
        pass
    
    def get_action(self, env: IEnv):
        legals = env.legal_move()
        #選択可能アクションが無いときは-1を返すことにする
        if(len(legals) == 0): return -1
        action = np.random.choice(legals)
        return action
    
    def get_first_board(self):
        pieces = np.arange(PIECE_LIMIT)
        np.random.shuffle(pieces)
        return pieces
        