from src.common import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Agent.DeepNash.resnet import PyramidModule


class DeepNashNetwork(nn.Module):
    def __init__(self, in_channels: int = 64, mid_channels: int = 82):
        super().__init__()
        
        self.p1 = PyramidModule(2,2, in_channels, mid_channels)
        
        self.pc = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        out_feature = (Config.board_shape[0] * Config.board_shape[1])**2
        in_feature = in_channels * Config.board_shape[0] * Config.board_shape[1]
        self.pl = nn.Linear(in_feature, out_feature)
        
        self.vc = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.vl = nn.Linear(in_feature, 1)
        
    def forward(self, obs: torch.Tensor, non_legal_move: torch.Tensor):
        out = self.p1.forward(obs)
        
        policy = self.pc.forward(out)
        policy = policy.reshape(policy.size(0), -1)
        policy = self.pl.forward(policy)
        policy = torch.where(non_legal_move, -1*torch.inf, policy)
        
        logit = policy.clone()
        
        policy = F.softmax(policy, dim=1)
        
        value = self.vc.forward(out)
        value = F.relu(value)
        value = value.reshape(value.size(0), -1)
        value = self.vl.forward(value)
        
        return policy,value,logit
        
        