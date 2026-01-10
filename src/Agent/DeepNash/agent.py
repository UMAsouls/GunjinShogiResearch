from src.Interfaces import IAgent
from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.cnn_network import DeepNashCnnNetwork

from src.Agent.DeepNash.ITensorBoard import ITensorBoard

from src.common import LogData, Config

import torch
import numpy as np

import GunjinShogiCore as GSC

def change_int_to_player(p:int):
    if(p == 1): return GSC.Player.PLAYER_ONE
    else: return GSC.Player.PLAYER_TWO
    
def change_int_to_erase(e:int):
    if(e == 1): return GSC.EraseFrag.BEF
    elif(e == 2): return GSC.EraseFrag.AFT
    else: return GSC.EraseFrag.BOTH


class DeepNashAgent(IAgent):
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        device: torch.device,
        tensor_board: ITensorBoard
    ):
        self.device = device
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device)
        self.network.eval() # 推論モード
        
        self.tensor_board = tensor_board
        
        self.deploy = True

    def load_state_dict(self, state_dict: dict):
        """学習済みモデルのパラメータをロードする"""
        self.network.load_state_dict(state_dict)
        self.network.eval() # 推論モードに設定
        
    def load_model(self, model_path: str):
        load_state_dict = torch.load(model_path)
        state_dict = {}
        for k, v in load_state_dict.items():
        # "_orig_mod." がついていたら削除する
            new_key = k.replace("_orig_mod.", "")
            state_dict[new_key] = v
        self.load_state_dict(state_dict)

    def get_action(self, env):
        obs_tensor = self.get_obs(env.get_current_player())
        obs_tensor = obs_tensor.unsqueeze(0).to(self.device)
        
        legals = env.legal_move()
        if len(legals) == 0:
            return -1
            
        non_legal_mask = np.ones((Config.board_shape_int**2), dtype=bool)
        non_legal_mask[legals] = False
        non_legal_tensor = torch.from_numpy(non_legal_mask).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy, _, _ = self.network(obs_tensor, non_legal_tensor)
            probs = policy
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
        return action
    
    def get_obs(self, player: GSC.Player):
        return self.tensor_board.get_board(player)
    
    def step(self, log:LogData, frag: GSC.BattleEndFrag):
        p = change_int_to_player(log.player)
        e = change_int_to_erase(log.erase)
        
        if(self.deploy):
            self.tensor_board.deploy_set(Config.first_dict[log.action], p)
        else:
            self.tensor_board.step(log.action, p, e)
            
        if(frag == GSC.BattleEndFrag.DEPLOY_END):
            self.deploy = False
            self.tensor_board.deploy_end()
        
    def reset(self):
        self.tensor_board.reset()
        self.deploy = True
    
    def get_first_board(self) -> np.ndarray:
        """初期配置の決定（現在はランダム）"""
        pieces = np.arange(Config.piece_limit)
        np.random.shuffle(pieces)
        return pieces
    
class DeepNashCnnAgent(DeepNashAgent):
    def __init__(self, in_channels, mid_channels, device, tensor_board, blocks = 7):
        super().__init__(in_channels, mid_channels, device, tensor_board)
        self.network = DeepNashCnnNetwork(in_channels, mid_channels, blocks).to(self.device)
        self.network.eval() # 推論モード