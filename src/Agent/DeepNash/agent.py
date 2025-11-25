import copy

from src.Interfaces import IAgent
from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer, Trajectory, Episode

from src.const import BOARD_SHAPE_INT

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class DeepNashAgent(IAgent):
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        device: torch.device,
        lr: float = 0.001,
        gamma: float = 0.99,
        eta: float = 0.1,  # R-NaDの正則化パラメータ (論文では0.2など)
        target_update_interval: int = 2000 # pi_reg を更新する間隔
    ):
        self.device = device
        self.gamma = gamma
        self.eta = eta
        self.target_update_interval = target_update_interval
        self.learn_step_counter = 0

        # Current Network (学習対象: pi)
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device)
        
        # Regularization Network (pi_reg / Target)
        # R-NaDにおいて「以前の戦略」を保持する重要な役割
        self.reg_network = copy.deepcopy(self.network).to(self.device)
        self.reg_network.eval() # 学習しない

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
    def learn(self, replay_buffer: ReplayBuffer, batch_size: int = 32):
        if len(replay_buffer) < batch_size:
            return
            
        self.network.train()
        episodes = replay_buffer.sample(batch_size)
        total_loss = 0
        
        for episode in episodes:
            states = torch.stack(episode.boards)
            actions = torch.tensor(episode.actions)
            rewards = torch.tensor(episode.rewards)
            behavior_policies = torch.stack(episode.policies)
            non_legals = torch.stack(episode.non_legals)
            
            # 1. 現在のネットワーク (pi) で計算
            policy_logits, values = self.network(states, non_legals)
            
            # 2. 正則化ネットワーク (pi_reg) で計算 (勾配不要)
            with torch.no_grad():
                reg_logits, _ = self.reg_network(states, non_legals)

            # 3. Reward Transform (R-NaDの核心)
            # 生の報酬ではなく、変換後の報酬を使ってV-traceを計算する
            transformed_rewards = self.reward_transform(
                rewards, policy_logits.detach(), reg_logits, actions
            )

            # 4. V-trace
            # ここでの target_policies は「現在の学習中のポリシー」を使う
            # (IMPALAではTarget Networkを使うこともあるが、R-NaDではpiを収束させるためpi自身を使うことが多い)
            target_policies = F.softmax(policy_logits.detach(), dim=1)
            
            vs, advantages = self.v_trace(
                behavior_policies, 
                target_policies, 
                actions, 
                transformed_rewards, # 変換済み報酬を使用
                values.detach(),     # Value Target
                self.gamma
            )
            
            # 5. Loss計算
            
            # Value Loss: Transformed Rewardに基づいた価値に近づける
            value_loss = F.mse_loss(values.squeeze(), vs)
            
            # Policy Loss
            log_probs = F.log_softmax(policy_logits, dim=1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            policy_loss = -(action_log_probs * advantages.detach()).mean()
            
            # Entropy
            probs = F.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            entropy_loss = -0.01 * entropy
            
            loss = policy_loss + 0.5 * value_loss + entropy_loss
            total_loss += loss

        self.optimizer.zero_grad()
        (total_loss / batch_size).backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=40.0)
        self.optimizer.step()
        
        # 6. Update Step (pi_reg の更新)
        # R-NaDにおける "Evolution of the population"
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            print("Update Regularization Policy (pi_reg) ...")
            self.reg_network.load_state_dict(self.network.state_dict())
        
    def v_trace(
        self, 
        behavior_policy_probs: torch.Tensor, 
        target_policy_probs: torch.Tensor, 
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
        advantage = torch.zeros_like(values)
        
        current_vs_plus_1 = 0.0 # vs_{t+1}
        
        # 時間を遡って計算
        for t in reversed(range(seq_len)):
            # vs_t = V(x_t) + delta_t + gamma * c_t * (vs_{t+1} - V(x_{t+1}))
            # 実際の実装では再帰的に計算する
            
            # 次のステップのVとの差分項
            next_v_diff = current_vs_plus_1 - next_values[t]
            
            vs[t] = values[t] + deltas[t] + gamma * cs[t] * next_v_diff
            
            # Advantage for policy update
            # A_t = (r_t + gamma * vs_{t+1}) - V(x_t)
            # ※論文によって定義が若干異なるが、ここでは一般的なQ-V形式を採用
            advantage[t] = rewards[t] + gamma * current_vs_plus_1 - values[t]
            
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
        log_probs = F.log_softmax(policy_logits, dim=1)
        reg_log_probs = F.log_softmax(reg_logits, dim=1)

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
            probs = F.softmax(policy_logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            
        return action
        
    
    def get_first_board(self):
        return super().get_first_board()