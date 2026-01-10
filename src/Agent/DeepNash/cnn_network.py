# network.py
from src.common import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

# ストライドなしのシンプルなResBlock
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DeepNashCnnNetwork(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 128, blocks = 7):
        super().__init__()
        
        # 入力層
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(mid_channels)
        
        # ResNetブロックを積み重ねる（サイズは変えない）
        # 5x3のような小規模盤面なら、層は浅くてOK（5〜10層程度）
        self.blocks = nn.Sequential(
            *[BasicBlock(mid_channels) for _ in range(blocks)]
        )
        
        out_feature = (Config.board_shape[0] * Config.board_shape[1])**2
        flat_features = mid_channels * Config.board_shape[0] * Config.board_shape[1]
        
        # Policy Head
        self.p_conv = nn.Conv2d(mid_channels, 32, kernel_size=1) # チャンネル圧縮
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * Config.board_shape[0] * Config.board_shape[1], out_feature)
        
        # Value Head
        self.v_conv = nn.Conv2d(mid_channels, 16, kernel_size=1)
        self.v_bn = nn.BatchNorm2d(16)
        self.v_fc1 = nn.Linear(16 * Config.board_shape[0] * Config.board_shape[1], 64)
        self.v_fc2 = nn.Linear(64, 1)
        
    def forward(self, obs: torch.Tensor, non_legal_move: torch.Tensor):
        # Backbone
        x = F.relu(self.bn_in(self.conv_in(obs)))
        x = self.blocks(x)
        
        # Policy
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.p_fc(p)
        
        # Masking (Illegal moves -> -inf)
        # cloneしないとIn-place操作で勾配計算がおかしくなることがあるので注意
        masked_logits = torch.where(non_legal_move, torch.tensor(-1e20, device=obs.device), policy_logits)
        policy = F.softmax(masked_logits, dim=1)
        
        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        value = self.v_fc2(v)
        
        # Learner側でNeuRDにつかうのは "masked_logits" (非合法手が潰されたlogit)
        return policy, value, masked_logits