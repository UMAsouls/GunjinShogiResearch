import copy
from math import log
import re

from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

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
        eta: float = 0.02,  # R-NaDの正則化パラメータ
        reg_update_interval: int = 1000, # pi_reg を更新する間隔(\Delta_m)
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

        #一個前のreg_network
        self.prev_reg_network = copy.deepcopy(self.network).to(self.device)
        self.prev_reg_network.eval() # 学習しない

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.losses = []
        self.log_q = []
        self.v_loss = []

    def get_current_network_state_dict(self) -> dict:
        """現在の学習済みネットワークのstate_dictを返す"""
        return self.target_network.state_dict()
    
    def get_regulized_target_policy(self, policy, r_policy, r_prev_policy, alpha:float) -> torch.Tensor:
        curr_log_probs = torch.log(policy/(r_policy + 1e-8)) * alpha
        prev_log_probs = torch.log(policy/(r_prev_policy + 1e-8)) * (1-alpha)

        return curr_log_probs + prev_log_probs

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
            players = episode.players.to(self.device).detach()

            n = self.learn_step_counter%self.reg_update_interval
            m = self.learn_step_counter//self.reg_update_interval
            alpha = min(1, 2*n/self.reg_update_interval)
            
            policy, values, logits = self.network(states, non_legals)
            with torch.no_grad():
                # 1. ターゲットネットワーク (pi_target) で計算
                target_policy, target_values, target_logits = self.target_network(states, non_legals)
                # 2. 正則化ネットワーク (pi_reg) で計算 (勾配不要)
                reg_policy, _, _ = self.reg_network(states, non_legals)

                reg_prev_policy, _, _ = self.prev_reg_network(states, non_legals)

                omega = self.get_regulized_target_policy(policy, reg_policy, reg_prev_policy, alpha)

                # 3. Reward Transform (R-NaDの核心)
                transformed_rewards = self.reward_transform(rewards, omega, actions, players)

                # 4. V-trace
                vs, qs = self.v_trace(
                    behavior_policy=behavior_policies,
                    target_policy=target_policy,
                    actions=actions,
                    rewards=transformed_rewards,
                    values=target_values,
                    players=players,
                    omega=omega, # log(pi/pi_reg) term
                    gamma=self.gamma,
                )
            
            # 5. Loss計算
            value_loss = F.mse_loss(values.squeeze(), vs)
            
            qs = clip(qs.detach(), self.c_clip_neurd)
            
            beta = 2.0
            policy_loss = self.policy_update(
                qs, policy, logits, players, self.c_clip_neurd,
                non_legals, beta,
            )

            loss = -policy_loss + value_loss

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
        target_policy: torch.Tensor,
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        values: torch.Tensor,
        players: torch.Tensor,
        omega: torch.Tensor, # log(pi/pi_reg) term for R-NaD
        gamma: float = 0.99,
        clip_rho_threshold: float = 1.0,
        clip_c_threshold: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = values.squeeze() # (SeqLen,)
        seq_len = values.shape[0]
        
        # 実際に選択されたアクションの確率を取り出す
        target_action_probs = target_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
        behavior_action_probs = behavior_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # V-trace target (vs) の計算を後ろから累積
        vs = torch.zeros_like(values)
        qs = torch.zeros_like(target_policy)

        xis = torch.zeros((2, seq_len+1), dtype=torch.float32)
        rhos = target_action_probs / (behavior_action_probs + 1e-8)

        next_values = torch.zeros((2, seq_len+1), dtype=torch.float32)
        n_rewards = torch.zeros((2, seq_len+1), dtype=torch.float32)

        vs = torch.zeros_like((2,seq_len+1), dtype=torch.float32)
        qs = torch.zeros_like((2,seq_len+1), dtype=torch.float32)

        #pi_theta_nとpi_mn_regのlogの差
        net_reg_log_diff = omega
        
        eta = self.eta
        # 時間を遡って計算
        for t in reversed(range(seq_len)):
            i = players[t]
            o = 1-i

            at = actions[t]

            xis[i][t] = 1
            xis[o][t] = rhos[t]*xis[o][t+1]
            
            rho_base = rhos[t]*xis[i][t+1]
            rho = torch.clamp(rho_base, max=clip_rho_threshold)
            c = torch.clamp(rho_base, max=clip_c_threshold)

            next_values[i][t] = values[t]
            next_values[o][t] = next_values[o][t+1]

            n_rewards[i][t] = 0
            n_rewards[o][t] = rewards[t][o] + rho_base*n_rewards[o][t+1]

            delta = rho*(rewards[t][i] + rho_base*n_rewards[i][t+1] - next_values[i][t+1] - values[t])

            vs[i][t] = values[t] + delta + c*(vs[i][t+1] - next_values[i][t+1])
            vs[o][t] = vs[o][t+1]

            qs[i][t] = -eta*net_reg_log_diff[t] + values[t].unsqueeze(0)
            qs[i][t][at] += (
                rewards[t][i] + net_reg_log_diff[t][at] + \
                rho_base*(n_rewards[i][t+1] + vs[i][t+1]) - values[t]            
            )/(behavior_action_probs[t] + 1e-8)
            
        if(qs.isnan().sum() > 0):
            print("nan detect")
            
        return vs, qs
    
    def reward_transform(
        self, 
        rewards: torch.Tensor, 
        omega: torch.Tensor, # log(pi/pi_reg) term
        actions: torch.Tensor,
        players: torch.Tensor
    ) -> torch.Tensor:
        """
        R-NaD Reward Transformation
        r_tilde = r - eta * log(pi(a|x) / pi_reg(a|x))
        """

        # 実際に選択したアクションの対数確率を取り出す
        omega_action = omega.gather(1, actions.unsqueeze(1)).squeeze(1)

        penalty = self.eta * omega_action

        play1 = torch.ones(penalty.shape, dtype=torch.int8)
        play1[players == 0] = -1
        play2 = -1*play1

        pe1 = penalty * play1
        pe2 = penalty * play2
        
        transformed_rewards = torch.t(rewards.clone())
        transformed_rewards[0] -= pe1
        transformed_rewards[1] -= pe2
        
        return transformed_rewards
    
    def make_reglogit_advantage(qs, policy, logits, non_legals, players, i, c_clip):
        ps = players == i

        qs = qs[i][ps]
        policy = policy[ps]
        logits = logits[ps]
        non_legals_i = non_legals[ps]

        legal_logits = logits[non_legals_i == 0]

        advantage = (qs - (qs*policy).sum(dim=1))
            
        # Policy Loss
        reg_logits = logits - legal_logits.mean(dim=1)

        return reg_logits, advantage, non_legals_i

    def policy_update(
            self,
            qs, policy, logits, players, c_clip_neurd,
            non_legals: torch.Tensor, beta
        ):
        sum_log_q = 0
        for i in range(2):

            reg_logits, adv, non_legals_i = self.make_reglogit_advantage(qs, policy, logits, non_legals, players, i, c_clip_neurd)

            with torch.no_grad():
                can_increase = reg_logits < beta
                can_decrease = reg_logits > -beta

                force_positive = torch.maximum(adv, 0.0)
                force_negative = torch.minimum(adv, 0.0)

                cliped_force = can_increase*force_positive + can_decrease*force_negative

            loss = reg_logits * cliped_force

            legal_loss = loss[non_legals_i == 0]

            sum_log_q += legal_loss.sum(dim=1).mean()

        return sum_log_q