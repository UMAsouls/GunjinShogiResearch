import copy

from src.Interfaces import IAgent
from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer, Trajectory, Episode

from src.const import BOARD_SHAPE_INT, PIECE_LIMIT

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import collections

import numpy as np

def clip(value,c):
    return torch.clip(value, -c, c)

class DeepNashAgent(IAgent):
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        device: torch.device,
        lr: float = 0.001,
        gamma: float = 0.99,
        eta: float = 0.1,  # R-NaDの正則化パラメータ (論文では0.2など)
        reg_update_interval: int = 2000, # pi_reg を更新する間隔
        gamma_ave: float = 0.5
    ):
        self.device = device
        self.gamma = gamma
        self.eta = eta
        self.reg_update_interval = reg_update_interval
        self.learn_step_counter = 0

        # Current Network (学習対象: pi)
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device)

        #target network
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval() # 学習しない
        
        # Regularization Network (pi_reg / Target)
        # R-NaDにおいて「以前の戦略」を保持する重要な役割
        self.reg_network = copy.deepcopy(self.network).to(self.device)
        self.reg_network.eval() # 学習しない

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.c_clip_neurd = 10

        self.gamma_ave = gamma_ave
    
    #Exponential Moving Average
    #currentとoldをalpha:1-alphaで融合し、新たなstate_dictを生み出す   
    def get_EMA_state_dict(self, current:nn.Module, old: nn.Module, alpha:float) -> dict:
        state_dict_current = current.state_dict()
        state_dict_old = old.state_dict()
        
        state_dict_new = {}
        for key in state_dict_current:
            param_crt = state_dict_current[key]
            param_old = state_dict_old[key]
            
            param_new = alpha*param_crt + (1-alpha)*param_old
            state_dict_new[key] = param_new
            
        return state_dict_new
        
    def learn(self, replay_buffer: ReplayBuffer, batch_size: int = 32):
        if len(replay_buffer) < batch_size:
            return
            
        self.network.train()
        episodes = replay_buffer.sample(batch_size)
        total_loss = 0
        
        for episode in episodes:
            #replay_bufferにあるやつに勾配を伝搬させないためにdetach()
            states = episode.boards.to(self.device).detach()
            actions = episode.actions.to(self.device).detach()
            rewards = episode.rewards.to(self.device).detach()
            behavior_policies = episode.policies.to(self.device).detach()
            non_legals = episode.non_legals.to(self.device).detach()
            
            policy, values, logits = self.network(states, non_legals)
            with torch.no_grad():
                # 1. ターゲットネットワーク (pi_target) で計算
                target_policy, target_values, _ = self.target_network(states, non_legals)
                # 2. 正則化ネットワーク (pi_reg) で計算 (勾配不要)
                reg_policy, _, _ = self.reg_network(states, non_legals)

                # 3. Reward Transform (R-NaDの核心)
                # 生の報酬ではなく、変換後の報酬を使ってV-traceを計算する
                transformed_rewards = self.reward_transform(
                    rewards, target_policy, reg_policy, actions
                )
                # 4. V-trace
                vs, advantages = self.v_trace(
                    behavior_policies, 
                    policy,
                    target_policy,
                    reg_policy,
                    actions, 
                    transformed_rewards, # 変換済み報酬を使用
                    target_values,     # Value Target
                    self.gamma
                )
            
            # 5. Loss計算
            
            # Value Loss: Transformed Rewardに基づいた価値に近づける
            value_loss = F.mse_loss(values.squeeze(), vs)
            
            qs = clip(advantages.detach(), self.c_clip_neurd)
            
            # Policy Loss
            l_theta = torch.where(non_legals, 0, logits)
            loss_base = l_theta * qs
            policy_loss = loss_base.sum(dim=1).mean()
            
            """# Entropy
            probs = F.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            entropy_loss = -0.01 * entropy
            """
            
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=40.0)
            self.optimizer.step()

            self.target_network.load_state_dict(
                self.get_EMA_state_dict(self.target_network,self.network, self.gamma_ave)
            )
        
            # 6. Update Step (pi_reg の更新)
            # R-NaDにおける "Evolution of the population"
            self.learn_step_counter += 1
            if self.learn_step_counter % self.reg_update_interval == 0:
                print("Update Regularization Policy (pi_reg) ...")
                self.reg_network.load_state_dict(self.network.state_dict())
            
    def delta_theta():
        pass
        
    def v_trace(
        self, 
        behavior_policy_probs: torch.Tensor,
        network_policy_logits: torch.Tensor,
        target_policy_probs: torch.Tensor,
        regnet_policy_logits: torch.Tensor,
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        gamma: float = 0.99,
        clip_rho_threshold: float = 1.0,
        clip_c_threshold: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        V-traceの実装 (IMPALA paper)
        Off-policyデータを補正して価値関数(v_s)とAdvantageを計算する
        """
        values = values.squeeze() # (SeqLen,)
        seq_len = values.shape[0]
        
        # 実際に選択されたアクションの確率を取り出す
        network_action_probs = network_policy_logits.gather(1, actions.unsqueeze(1)).squeeze() 
        target_action_probs = target_policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
        behavior_action_probs = behavior_policy_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 重要度重み (Importance Sampling Ratio)
        # rho = min(bar_rho, pi(a|x) / mu(a|x))
        rhos = target_action_probs / (behavior_action_probs + 1e-8)
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        cs = torch.clamp(rhos, max=clip_c_threshold)
        
        # 次の状態の価値 (V(x_{t+1}))。最後は0と仮定
        next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])
        
        # デルタ (TD誤差のようなもの)
        # delta_t = rho_t * (r_t + gamma * V(x_{t+1}) - V(x_t))
        deltas = clipped_rhos * (rewards + gamma * next_values - values)
        
        # V-trace target (vs) の計算を後ろから累積
        vs = torch.zeros_like(values)
        advantage = torch.zeros_like(network_policy_logits)
        
        current_vs_plus_1 = 0.0 # vs_{t+1}

        #pi_theta_nとpi_mn_regのlogの差
        net_reg_log_diff = torch.log(network_policy_logits+1e-8) - torch.log(regnet_policy_logits+1e-8)
        
        eta = 0.01
        # 時間を遡って計算
        for t in reversed(range(seq_len)):
            # vs_t = V(x_t) + delta_t + gamma * c_t * (vs_{t+1} - V(x_{t+1}))
            # 実際の実装では再帰的に計算する
            
            # 次のステップのVとの差分項
            next_v_diff = current_vs_plus_1 - next_values[t]
            
            vs[t] = values[t] + deltas[t] + cs[t] * next_v_diff
            
            # Advantage for policy update
            # A_t = (r_t + gamma * vs_{t+1}) - V(x_t)
            # ※論文によって定義が若干異なるが、ここでは一般的なQ-V形式を採用
            #advantage[t] = rewards[t] + gamma * current_vs_plus_1 - values[t]

            #advantage = Q(a_t)
            advantage[t] = -eta*net_reg_log_diff[t] + values[t].unsqueeze(0)
            #選択されたactionに対する補正
            advantage[t][actions[t]] += (rewards[t] + net_reg_log_diff[t][actions[t]] + rhos[t] * current_vs_plus_1 - values[t])/behavior_action_probs[t]
            
            current_vs_plus_1 = vs[t]
            
        return vs, advantage
    
    def reward_transform(
        self, 
        rewards: torch.Tensor, 
        policy_logits: torch.Tensor, 
        reg_logits: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        R-NaD Reward Transformation
        r_tilde = r - eta * log(pi(a|x) / pi_reg(a|x))
                = r - eta * (log_pi(a|x) - log_pi_reg(a|x))
        """
        # Log Softmaxで対数確率を取得
        log_probs = torch.log(policy_logits)
        reg_log_probs = torch.log(reg_logits)

        # 実際に選択したアクションの対数確率を取り出す
        # actions: (Batch,) -> (Batch, 1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        reg_action_log_probs = reg_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # 報酬の変換
        # pi(a|x) / pi_reg(a|x) が大きい（今の戦略の方が確率が高い）と、
        # log(...) は正になり、報酬が減らされる（=以前の戦略から離れることへのペナルティ）
        penalty = self.eta * (action_log_probs - reg_action_log_probs)
        
        # penaltyは勾配計算には含めない（報酬として扱うため）
        transformed_rewards = rewards - penalty.detach()
        
        return transformed_rewards
        
    def get_action(self, env):
        self.network.eval()
        obs_tensor = env.get_tensor_board_current()
        
        legals = env.legal_move()
        if len(legals) == 0:
            return -1
            
        non_legal_mask = np.ones((BOARD_SHAPE_INT**2), dtype=bool)
        non_legal_mask[legals] = False
        non_legal_tensor = torch.from_numpy(non_legal_mask).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, _ = self.network(obs_tensor, non_legal_tensor)
            probs = policy_logits
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
        return action
        
    
    def get_first_board(self) -> np.ndarray:
        """初期配置の決定（現在はランダム）"""
        pieces = np.arange(PIECE_LIMIT)
        np.random.shuffle(pieces)
        return pieces