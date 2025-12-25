
from dataclasses import dataclass
from math import e

import torch
import numpy as np

from src.common import Config

import GunjinShogiCore as GSC

import random
from collections import deque

import multiprocessing as mp

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
        self.boards: torch.Tensor = torch.zeros((max_step, *board_tensor_shape), dtype=torch.float32)
        self.actions: torch.Tensor = torch.zeros(max_step, dtype=torch.int32)
        self.rewards: torch.Tensor = torch.zeros((max_step,2),dtype=torch.float32)
        self.policies: torch.Tensor = torch.zeros((max_step, Config.board_shape_int**2), dtype=torch.float32)
        self.non_legals: torch.Tensor = torch.zeros((max_step, Config.board_shape_int**2), dtype=torch.bool)
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

    def set_reward_all(self, p1_reward:float, p2_reward:float):
        # 一旦ゼロクリア（初期化でゼロなら不要だが念のため）
        self.rewards.fill_(0)

        # 最終ステップ (head-1) のみに設定
        last_step = self.head - 1
        if last_step >= 0:
            self.rewards[last_step, 0] = p1_reward
            self.rewards[last_step, 1] = p2_reward


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
        
        self.head = 0
        self.size = size
        self.max_step = max_step
        self.length = 0
        
        self.board_shape = board_shape
        
        self.boards = torch.zeros((size, max_step, *board_shape), dtype=torch.float32)
        self.actions = torch.zeros((size, max_step), dtype=torch.int32)
        self.rewards = torch.zeros((size, max_step, 2), dtype=torch.float32)    
        self.non_legals = torch.zeros((size, max_step, Config.board_shape_int**2), dtype=torch.bool)
        self.policies = torch.zeros((size, max_step, Config.board_shape_int**2), dtype=torch.bool)
        self.players = torch.zeros((size, max_step), dtype=torch.int32)
        self.mask = torch.zeros((size, max_step), dtype=torch.bool)
        self.t_effective = torch.zeros((size), dtype=torch.int32)
        
        self.mp: bool = False    

    def add(self, episode: Episode) -> bool:
        if(self.mp): return False
        
        t = episode.t_effective
        
        self.boards[self.head, :t] = episode.boards
        self.actions[self.head, :t] = episode.actions
        self.rewards[self.head, :t] = episode.rewards
        self.non_legals[self.head, :t] = episode.non_legals
        self.policies[self.head, :t] = episode.policies
        self.players[self.head, :t] = episode.players
        self.mask[self.head, :episode.t_effective] = True
        self.t_effective[self.head] = episode.t_effective
        
        self.head = (self.head + 1) % self.size
        self.length = min(self.length + 1, self.size)
            
        return True

    def sample(self, batch_size: int) -> MiniBatch:
        if(self.mp):
            length = self.length.value
        else:
            length = self.length
        
        random_indices = torch.randint(0, length, (batch_size,))

        max_t_effective = self.max_step

        minibatch = MiniBatch(
            torch.zeros((batch_size, max_t_effective, *self.board_shape), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective), dtype=torch.int32),
            torch.zeros((batch_size, max_t_effective, 2), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective, Config.board_shape_int**2), dtype=torch.float32),
            torch.zeros((batch_size, max_t_effective, Config.board_shape_int**2), dtype=torch.bool),
            torch.zeros((batch_size, max_t_effective), dtype=torch.int32),
            torch.zeros((batch_size, max_t_effective), dtype=torch.bool),
            torch.zeros((batch_size), dtype=torch.int32),
            max_t_effective
        )
        
        t_effective = self.t_effective[random_indices]
        
        for i in range(batch_size):
            idx = random_indices[i]            # Buffer上のインデックス
            eff_len = t_effective[i].item() # そのエピソードの有効長さ
            
            # スライスを使ってコピー
            minibatch.boards[i, :eff_len] = self.boards[idx, :eff_len]
            minibatch.actions[i, :eff_len] = self.actions[idx, :eff_len]
            minibatch.rewards[i, :eff_len] = self.rewards[idx, :eff_len]
            minibatch.non_legals[i, :eff_len] = self.non_legals[idx, :eff_len]
            minibatch.policies[i, :eff_len] = self.policies[idx, :eff_len]
            minibatch.players[i, :eff_len] = self.players[idx, :eff_len]
            minibatch.mask[i, :eff_len] = self.mask[idx, :eff_len]
            
        
        return minibatch
    
    def mp_set(self):
        self.boards.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.non_legals.share_memory_()
        self.players.share_memory_()
        self.mask.share_memory_()
        self.t_effective.share_memory_()
        
        self.head = mp.Value('i', self.head)
        self.length = mp.Value('i', self.length)
        
        self.mp = True
        
    def __len__(self):
        if (self.mp):
            return self.length.value
        else:
            return self.length
    
