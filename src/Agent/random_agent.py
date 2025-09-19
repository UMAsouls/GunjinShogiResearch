from src.Interfaces import IAgent, IEnv
from src.const import PIECE_LIMIT

import torch

class RandomAgent(IAgent):
    def __init__(self):
        pass
    
    def get_action(self, env: IEnv):
        legals = env.legal_move()
        idx = torch.multinomial(torch.ones(legals.shape), num_samples=1, replacement=True)
        action = legals[idx]
        return action.item()
    
    def get_first_board(self):
        pieces = torch.arange(PIECE_LIMIT)
        rand = torch.randperm(pieces.shape[0])
        return pieces[rand]
        