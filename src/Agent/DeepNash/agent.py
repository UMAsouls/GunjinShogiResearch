from src.Interfaces import IAgent
from src.Agent.DeepNash.network import DeepNashNetwork

from src.const import BOARD_SHAPE_INT, PIECE_LIMIT

import torch
import numpy as np

class DeepNashAgent(IAgent):
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        device: torch.device,
    ):
        self.device = device
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device)
        self.network.eval() # 推論モード

    def load_state_dict(self, state_dict: dict):
        """学習済みモデルのパラメータをロードする"""
        self.network.load_state_dict(state_dict)
        self.network.eval() # 推論モードに設定
        
    def load_model(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def get_action(self, env):
        obs_tensor = env.get_tensor_board_current().unsqueeze(0).to(self.device)
        
        legals = env.legal_move()
        if len(legals) == 0:
            return -1
            
        non_legal_mask = np.ones((BOARD_SHAPE_INT**2), dtype=bool)
        non_legal_mask[legals] = False
        non_legal_tensor = torch.from_numpy(non_legal_mask).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy, _, _ = self.network(obs_tensor, non_legal_tensor)
            probs = policy
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
        return action
        
    
    def get_first_board(self) -> np.ndarray:
        """初期配置の決定（現在はランダム）"""
        pieces = np.arange(PIECE_LIMIT)
        np.random.shuffle(pieces)
        return pieces