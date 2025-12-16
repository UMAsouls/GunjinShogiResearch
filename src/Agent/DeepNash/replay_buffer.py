
from dataclasses import dataclass
from math import e

import torch
import numpy as np

from src.const import BOARD_SHAPE_INT, BOARD_SHAPE
import GunjinShogiCore as GSC

import random
from collections import deque

@dataclass
class Trajectory:
    board: torch.Tensor
    action: int
    reward: float
    policy: torch.Tensor
    player: GSC.Player
    non_legal: torch.Tensor
    
class Episode:
    def __init__(self, board_tensor_shape:tuple, max_step = 2000):
        sample_b = torch.zeros(board_tensor_shape, dtype=torch.float32)
        
        self.boards : torch.Tensor = sample_b.unsqueeze(0).repeat(max_step,1,1,1)
        self.actions: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)
        self.rewards: torch.Tensor = torch.zeros((max_step,2),dtype=torch.float32)
        self.policies: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.float32)
        self.non_legals: torch.Tensor = torch.zeros((max_step, BOARD_SHAPE_INT**2), dtype=torch.bool)
        self.players: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)

        self.t_effective:int = 0
        self.head = 0
        
    def episode_end(self):
        self.boards = self.boards[:self.head]
        self.actions = self.actions[:self.head]
        self.rewards = self.rewards[:self.head]
        self.policies = self.policies[:self.head]
        self.non_legals = self.non_legals[:self.head]
        self.players = self.players[:self.head]
        
    def add_step(self, trac:Trajectory):
        #tensorはcpuに保存
        self.boards[self.head] = trac.board.cpu()
        self.actions[self.head] = trac.action
        self.rewards[self.head] = trac.reward
        self.policies[self.head] = trac.policy.cpu()
        self.non_legals[self.head] = trac.non_legal.cpu()
        self.players[self.head] = 0 if trac.player == GSC.Player.PLAYER_ONE else 1
        
        self.t_effective += 1
        self.head += 1
        
    def set_reward(self, reward:float):
        # 一旦ゼロクリア（初期化でゼロなら不要だが念のため）
        self.rewards.fill_(0)

        # 最終ステップ (head-1) のみに設定
        last_step = self.head - 1
        if last_step >= 0:
            self.rewards[last_step, 0] = reward
            self.rewards[last_step, 1] = -1 * reward


@dataclass
class MiniBatch:
    boards: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    policies: torch.Tensor
    non_legals: torch.Tensor
    players: torch.Tensor
    mask: torch.Tensor
    t_effective: torch.Tensor
    max_t_effective: torch.Tensor



class ReplayBuffer:
    def __init__(self, size:int = 10000, max_step:int = 1000, board_shape:tuple = [58,6,9]):
        self.buffer = deque(maxlen=size)
        
        self.head = 0
        self.size = size
        self.max_step = max_step
        self.length = 0
        
        self.board_shape = board_shape

    def add(self, episode: Episode):
        self.buffer.append(episode)
        
        self.head = (self.head + 1) % self.size
        self.length = min(self.length + 1, self.size)
        

    def sample(self, batch_size: int) -> MiniBatch:
        t_effective = torch.zeros((batch_size), dtype=torch.int32)

        episodes = random.sample(self.buffer, batch_size)

        for i in range(batch_size):
            t_effective[i] = episodes[i].t_effective

        max_t_effective = self.max_step

        minibatch = MiniBatch(
            torch.zeros((batch_size, max_t_effective, *self.board_shape), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective), dtype=torch.int32),
            torch.zeros((batch_size, max_t_effective, 2), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective, BOARD_SHAPE_INT**2), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective, BOARD_SHAPE_INT**2), dtype=torch.bool),
            torch.zeros((batch_size, max_t_effective), dtype=torch.int32),
            torch.zeros((batch_size, max_t_effective), dtype=torch.bool),
            t_effective,
            max_t_effective
        )

        
        for i in range(batch_size):
            t = t_effective[i]
            episode = episodes[i]
            
            minibatch.boards[i, :t] = episode.boards
            minibatch.actions[i, :t] = episode.actions
            minibatch.rewards[i, :t] = episode.rewards
            minibatch.policies[i, :t] = episode.policies
            minibatch.non_legals[i, :t] = episode.non_legals
            minibatch.players[i, :t] = episode.players
            minibatch.mask[i, :t] = torch.ones(t, dtype=torch.bool)
            
        
        return minibatch
        
    def __len__(self):
        return self.length
    
