import copy

from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def clip(value, c):
    return torch.clip(value, -c, c)

class DeepNashLearner:
    def __init__(
        self, 
        in_channels: int, 
        mid_channels: int, 
        device: torch.device,
        lr: float = 0.001,
        gamma: float = 0.99,
        eta: float = 0.1,  # R-NaDの正則化パラメータ
        reg_update_interval: int = 1000, # pi_reg を更新する間隔
        gamma_ave: float = 0.5
    ):
        self.device = device
        self.gamma = gamma
        self.eta = eta
        self.reg_update_interval = reg_update_interval
        self.learn_step_counter = 0
        self.gamma_ave = gamma_ave
        self.c_clip_neurd = 10000

        # Current Network (学習対象: pi)
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device)

        # Target network
        self.target_network = copy.deepcopy(self.network).to(self.device)
        self.target_network.eval() # 学習しない
        
        # Regularization Network (pi_reg / Target)
        self.reg_network = copy.deepcopy(self.network).to(self.device)
        self.reg_network.eval() # 学習しない

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.losses = []
        self.log_q = []
        self.v_loss = []

    def get_current_network_state_dict(self) -> dict:
        """現在の学習済みネットワークのstate_dictを返す"""
        return self.target_network.state_dict()

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
        
    def learn(self, replay_buffer: ReplayBuffer, batch_size: int = 32, loss_print_path:str = "loss"):
        if len(replay_buffer) < batch_size:
            return
            
        self.network.train()
        episodes = replay_buffer.sample(batch_size)
        
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
                target_policy, target_values, target_logits = self.target_network(states, non_legals)
                # 2. 正則化ネットワーク (pi_reg) で計算 (勾配不要)
                reg_policy, _, _ = self.reg_network(states, non_legals)

                # 3. Reward Transform (R-NaDの核心)
                transformed_rewards = self.reward_transform(
                    rewards, policy, reg_policy, actions
                )
                # 4. V-trace
                vs, advantages = self.v_trace(
                    behavior_policies, 
                    policy,
                    target_policy,
                    reg_policy,
                    actions, 
                    transformed_rewards,
                    target_values,
                    self.gamma
                )
            
            # 5. Loss計算
            value_loss = F.mse_loss(values.squeeze(), vs)
            
            qs = clip(advantages.detach(), self.c_clip_neurd)
            
            # Policy Loss
            l_theta = torch.where(non_legals, 0, logits)
            loss_base = l_theta * qs
            logit_q = loss_base.sum(dim=1).mean()
            
            loss = -logit_q + value_loss
            
            self.losses.append(loss.item())
            self.log_q.append(logit_q.item())
            self.v_loss.append(value_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=40.0)
            self.optimizer.step()

            self.target_network.load_state_dict(
                self.get_EMA_state_dict(self.target_network,self.network, self.gamma_ave)
            )
        
            # 6. Update Step (pi_reg の更新)
            self.learn_step_counter += 1
            if self.learn_step_counter % self.reg_update_interval == 0:
                print("Update Regularization Policy (pi_reg) ...")
                self.reg_network.load_state_dict(self.network.state_dict())
        
        fig = plt.figure(figsize=(8,5))
        fig.suptitle("loss")
        
        plt.plot(self.losses, label="合計loss")
        plt.xlabel("epoc")
        plt.legend()
        plt.grid()
        plt.savefig(f"{loss_print_path}/all_loss.png", format="png")
        plt.cla()
        
        plt.plot(self.log_q, label="p_log × Qの平均")
        plt.xlabel("epoc")
        plt.legend()
        plt.grid()
        plt.savefig(f"{loss_print_path}/logit_q.png", format="png")
        plt.cla()
        
        plt.plot(self.v_loss, label = "v_loss")
        plt.xlabel("epoc")
        plt.legend()
        plt.grid()
        plt.savefig(f"{loss_print_path}/v_loss.png", format="png")
        plt.cla()
        
        plt.clf()
        plt.close()
    
    def v_trace(
        self, 
        behavior_policy: torch.Tensor,
        network_policy: torch.Tensor,
        target_policy: torch.Tensor,
        regnet_policy: torch.Tensor,
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        gamma: float = 0.99,
        clip_rho_threshold: float = 1.0,
        clip_c_threshold: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = values.squeeze() # (SeqLen,)
        seq_len = values.shape[0]
        
        # 実際に選択されたアクションの確率を取り出す
        network_action_probs = network_policy.gather(1, actions.unsqueeze(1)).squeeze() 
        target_action_probs = target_policy.gather(1, actions.unsqueeze(1)).squeeze()
        behavior_action_probs = behavior_policy.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 重要度重み (Importance Sampling Ratio)
        rhos = target_action_probs / (behavior_action_probs + 1e-8)
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        cs = torch.clamp(rhos, max=clip_c_threshold)
        
        # 次の状態の価値 (V(x_{t+1}))。最後は0と仮定
        next_values = torch.cat([values[1:], torch.tensor([0.0], device=self.device)])
        
        # デルタ (TD誤差のようなもの)
        deltas = clipped_rhos * (rewards + gamma * next_values - values)
        
        # V-trace target (vs) の計算を後ろから累積
        vs = torch.zeros_like(values)
        advantage = torch.zeros_like(network_policy)
        
        current_vs_plus_1 = 0.0 # vs_{t+1}

        #pi_theta_nとpi_mn_regのlogの差
        net_reg_log_diff = torch.log(network_policy+1e-8) - torch.log(regnet_policy+1e-8)
        
        eta = self.eta
        # 時間を遡って計算
        for t in reversed(range(seq_len)):
            next_v_diff = current_vs_plus_1 - next_values[t]
            
            vs[t] = values[t] + deltas[t] + cs[t] * next_v_diff
            
            advantage[t] = -eta*net_reg_log_diff[t] + values[t].unsqueeze(0)
            #選択されたactionに対する補正
            advantage[t][actions[t]] += (rewards[t] + net_reg_log_diff[t][actions[t]] + rhos[t] * current_vs_plus_1 - values[t])/behavior_action_probs[t]
            
            current_vs_plus_1 = vs[t]
            
        if(advantage.isnan().sum() > 0):
            print("nan detect")
            
        return vs, advantage
    
    def reward_transform(
        self, 
        rewards: torch.Tensor, 
        policy: torch.Tensor, 
        reg_policy: torch.Tensor, 
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        R-NaD Reward Transformation
        r_tilde = r - eta * log(pi(a|x) / pi_reg(a|x))
        """
        log_probs = torch.log(policy + 1e-8)
        reg_log_probs = torch.log(reg_policy + 1e-8)

        # 実際に選択したアクションの対数確率を取り出す
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        reg_action_log_probs = reg_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        penalty = self.eta * (action_log_probs - reg_action_log_probs)
        
        transformed_rewards = rewards - penalty.detach()
        
        return transformed_rewards