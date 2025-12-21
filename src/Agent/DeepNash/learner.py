import copy
from math import log
import os

from src.Agent.DeepNash.network import DeepNashNetwork
from src.Agent.DeepNash.replay_buffer import ReplayBuffer, MiniBatch

import matplotlib.pyplot as plt
import japanize_matplotlib

from dataclasses import dataclass

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import gc

def clip(value, c):
    return torch.clip(value, -c, c)

@torch.jit.script
class VtraceFirst:
    def __init__(self, xis_f: torch.Tensor, next_values_f: torch.Tensor, 
                 n_rewards_f: torch.Tensor, vs_f: torch.Tensor, qs_f: torch.Tensor):
        self.xis_f = xis_f
        self.next_values_f = next_values_f
        self.n_rewards_f = n_rewards_f
        self.vs_f = vs_f
        self.qs_f = qs_f
        
@torch.jit.script
def v_trace(
    behavior_policy: torch.Tensor,
    target_policy: torch.Tensor,
    actions: torch.Tensor, 
    rewards: torch.Tensor, 
    values: torch.Tensor,
    players: torch.Tensor,
    omega: torch.Tensor, # log(pi/pi_reg) term for R-NaD
    mask: torch.Tensor,
    eta: float,
    device:torch.device,
    vtracefirst: VtraceFirst,
    clip_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, VtraceFirst]:
    values = values.squeeze() # (SeqLen,)
    seq_len:int = values.shape[1]
    batch_size:int = values.shape[0]
        
    # 実際に選択されたアクションの確率を取り出す
    target_action_probs = target_policy.gather(2, actions.unsqueeze(2)).squeeze(2)
    behavior_action_probs = behavior_policy.gather(2, actions.unsqueeze(2)).squeeze(2)
    
    # V-trace target (vs) の計算を後ろから累積

    xis = torch.zeros((batch_size, seq_len+1, 2), dtype=torch.float32, device=device)
    rhos = target_action_probs / (behavior_action_probs + 1e-8)

    next_values = torch.zeros((batch_size, seq_len+1, 2), dtype=torch.float32, device=device)
    n_rewards = torch.zeros((batch_size, seq_len+1, 2), dtype=torch.float32, device=device)

    vs = torch.zeros((batch_size, seq_len+1, 2), dtype=torch.float32, device=device)
    qs = torch.zeros((batch_size, seq_len+1, 2, target_policy.shape[2]), dtype=torch.float32, device=device)


    b_idx = torch.arange(batch_size, device=device)

    m_sum = mask.sum(dim=1)
    xis[b_idx,m_sum,:] = vtracefirst.xis_f[b_idx,:]
    next_values[b_idx,m_sum,:] = vtracefirst.next_values_f[b_idx,:]
    n_rewards[b_idx,m_sum,:] = vtracefirst.n_rewards_f[b_idx,:]
    vs[b_idx,m_sum,:] = vtracefirst.vs_f[b_idx,:]
    qs[b_idx,m_sum,:,:] = vtracefirst.qs_f[b_idx,:,:]


    #pi_theta_nとpi_mn_regのlogの差
    net_reg_log_diff = omega
    
    ts = torch.arange(seq_len, device=device)
    ts = torch.flip(ts, dims=(0,))
    
    # 時間を遡って計算
    for t in ts:
        m = mask[:,t]
        
        i = players[:,t]
        o = 1-i

        at = actions[:,t]

        xis[b_idx,t,i] = 1
        xis[b_idx,t,o] = rhos[:,t]*xis[b_idx,t+1,o] * m
        
        rho_base = rhos[:,t]*xis[b_idx,t+1,i] * m
        rho = torch.clamp(rho_base, max=clip_rho_threshold) * m
        c = torch.clamp(rho_base, max=clip_c_threshold) * m

        next_values[b_idx,t,i] = values[:,t]*m
        next_values[b_idx,t,o] = next_values[b_idx,t+1,o]*m

        n_rewards[b_idx,t,i] = 0
        n_rewards[b_idx,t,o] = (rewards[b_idx,t,o] + rhos[:,t]*n_rewards[b_idx,t+1,o])*m

        delta = rho*(rewards[b_idx,t,i] + rhos[:,t]*n_rewards[b_idx,t+1,i] - next_values[b_idx,t+1,i] - values[:,t])*m

        vs[b_idx,t,i] = values[b_idx,t] + delta + c*(vs[b_idx,t+1,i] - next_values[b_idx,t+1,i])*m
        vs[b_idx,t,o] = vs[b_idx,t+1,o]*m

        qs[b_idx,t,i] = (net_reg_log_diff[:,t]*(-eta) + values[:,t].unsqueeze(1))*m.unsqueeze(1)
        term = m * (
            rewards[b_idx, t, i] + 
            net_reg_log_diff[b_idx, t, at] * eta + 
            rhos[:,t] * (n_rewards[b_idx, t+1, i] + vs[b_idx, t+1, i]) - 
            values[:, t]
        ) / (behavior_action_probs[:, t] + 1e-8)
        
        qs[b_idx, t, i, at] += term

    vtracefirst.xis_f = xis[:,0,:]
    vtracefirst.next_values_f = next_values[:,0,:]
    vtracefirst.n_rewards_f = n_rewards[:,0,:]
    vtracefirst.vs_f = vs[:,0,:]
    vtracefirst.qs_f = qs[:,0,:,:]
    
        
    return vs[:,:seq_len,:], qs[:,:seq_len,:,:], vtracefirst
    

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
        gamma_ave: float = 0.005,
        huber_delta: float = 10
    ):
        self.device = device
        self.gamma = gamma
        self.eta = eta
        self.reg_update_interval = reg_update_interval
        self.learn_step_counter = 0
        self.gamma_ave = gamma_ave
        self.c_clip_neurd = 10000
        self.c_clip_grad = 10000
        self.huber_delta = huber_delta

        # Current Network (学習対象: pi)
        self.network = DeepNashNetwork(in_channels, mid_channels).to(self.device, memory_format=torch.channels_last)

        # Target network
        self.target_network = copy.deepcopy(self.network).to(self.device, memory_format=torch.channels_last)
        self.target_network.eval() # 学習しない
        
        # Regularization Network (pi_reg / Target)
        self.reg_network = copy.deepcopy(self.network).to(self.device, memory_format=torch.channels_last)
        self.reg_network.eval() # 学習しない

        #一個前のreg_network
        self.prev_reg_network = copy.deepcopy(self.network).to(self.device, memory_format=torch.channels_last)
        self.prev_reg_network.eval() # 学習しない
        
        self.network:torch.nn.Module = torch.compile(self.network, backend="cudagraphs")
        self.target_network:torch.nn.Module = torch.compile(self.target_network, backend="cudagraphs")
        self.reg_network:torch.nn.Module = torch.compile(self.reg_network, backend="cudagraphs")
        self.prev_reg_network:torch.nn.Module = torch.compile(self.prev_reg_network, backend="cudagraphs")
    
        

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.file_inited = False
        self.losses = []
        self.log_q = []
        self.v_loss = []

    def get_current_network_state_dict(self) -> dict:
        """現在の学習済みネットワークのstate_dictを返す"""
        return self.target_network.state_dict()
    
    def get_regulized_target_policy(self, policy, r_policy, r_prev_policy, alpha:float) -> torch.Tensor:
        curr_log_probs = torch.log((policy + 1e-8)/(r_policy + 1e-8) ) * alpha
        prev_log_probs = torch.log((policy+ 1e-8)/(r_prev_policy + 1e-8)) * (1-alpha)

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
    
    def get_v_q(self, 
                flat_states, flat_non_legals, 
                rewards, actions, players, masks, 
                policy, behavior_policies, alpha:float,
                vtracefirst: VtraceFirst
            ):
        # 1. ターゲットネットワーク (pi_target) で計算
        flat_target_policy, flat_target_values, _ = self.target_network(flat_states, flat_non_legals)
        target_policy = flat_target_policy.reshape(rewards.shape[0], rewards.shape[1], -1)
        target_values = flat_target_values.reshape(rewards.shape[0], rewards.shape[1], 1)
            
        # 2. 正則化ネットワーク (pi_reg) で計算 (勾配不要)
        flat_reg_policy, _, _ = self.reg_network(flat_states, flat_non_legals)
        reg_policy = flat_reg_policy.reshape(rewards.shape[0], rewards.shape[1], -1)

        flat_reg_prev_policy, _, _ = self.prev_reg_network(flat_states, flat_non_legals)
        reg_prev_policy = flat_reg_prev_policy.reshape(rewards.shape[0], rewards.shape[1], -1)

        omega = self.get_regulized_target_policy(policy, reg_policy, reg_prev_policy, alpha)

        # 3. Reward Transform (R-NaDの核心)
        transformed_rewards = self.reward_transform(rewards, omega, actions, players)

        # 4. V-trace
        vs, qs, vtracefirst = v_trace(
            behavior_policy=behavior_policies,
            target_policy=target_policy,
            actions=actions,
            rewards=transformed_rewards,
            values=target_values,
            players=players,
            mask=masks,
            eta=self.eta,
            device=self.device,
            vtracefirst=vtracefirst,
            omega=omega, # log(pi/pi_reg) term
        )
        
        return vs, qs, vtracefirst
    
    def get_loss(self, minibatch: MiniBatch, vtracefirst: VtraceFirst, start:int, end:int):
        policy_loss = torch.Tensor([0.0]).to(self.device)
        value_loss = torch.Tensor([0.0]).to(self.device)
        loss = torch.Tensor([0.0]).to(self.device)

        n = self.learn_step_counter%self.reg_update_interval
        m = self.learn_step_counter//self.reg_update_interval
        alpha = min(1, 2*n/self.reg_update_interval)
        
        #replay_bufferにあるやつに勾配を伝搬させないためにdetach()
        
        with torch.no_grad():
            states = minibatch.boards[:,start:end].to(self.device)
            actions = minibatch.actions[:,start:end].to(self.device)
            rewards = minibatch.rewards[:,start:end].to(self.device)
            behavior_policies = minibatch.policies[:,start:end].to(self.device)
            non_legals = minibatch.non_legals[:,start:end].to(self.device)
            players = minibatch.players[:,start:end].to(self.device)
            masks = minibatch.mask[:,start:end].to(self.device)
        
            flat_states = states.reshape(-1, *states.shape[2:])
            flat_non_legals = non_legals.reshape(-1, *non_legals.shape[2:])

            batch_size = states.shape[0]
        
            
        flat_policy, flat_values, flat_logits = self.network(flat_states, flat_non_legals)
        policy = flat_policy.reshape(states.shape[0], states.shape[1], -1)
        values = flat_values.reshape(states.shape[0], states.shape[1], 1)
        logits = flat_logits.reshape(states.shape[0], states.shape[1], -1)
        
        with torch.no_grad():
            vs, qs, vtracefirst = self.get_v_q(
                flat_states, flat_non_legals,
                rewards, actions, players, masks,
                policy, behavior_policies, alpha,
                vtracefirst
            )
            
        del flat_states, flat_non_legals, flat_policy, flat_values, flat_logits
            
        # 5. Loss計算
        t_values = values.squeeze()
        t_values1 = t_values[(players==0)*masks]
        t_values2 = t_values[(players==1)*masks]
        
        c_clip_vs = 100000
        vs = clip(vs.detach(), c_clip_vs)
        vs1 = vs[(players==0)*masks, 0]
        vs2 = vs[(players==1)*masks, 1]
            
        value_loss = F.huber_loss(t_values1, vs1, delta=self.huber_delta) + F.huber_loss(t_values2, vs2, delta=self.huber_delta)
            
        qs = clip(qs.detach(), self.c_clip_neurd)
            
        beta = 2.0
        policy_loss += self.policy_update(
            qs, policy, logits, players, masks, 
            self.c_clip_neurd, non_legals, beta,
        )
        
        policy_loss = policy_loss/batch_size
        
        loss = -policy_loss + value_loss

        return policy_loss, value_loss, loss, vtracefirst
        
    def learn(
            self, replay_buffer: ReplayBuffer, 
            batch_size: int = 32, fixed_game_size:int = 200, accumration:int = 4,
            loss_print_path:str = "loss"
        ):
        if len(replay_buffer) < batch_size:
            return
        torch.compiler.cudagraph_mark_step_begin()
        self.network.train()
        
        policy_loss = 0
        value_loss = 0
        loss = 0

        for a in range(accumration):
            minibatch = replay_buffer.sample(batch_size)
            max_game_size = minibatch.max_t_effective

            vtracefirst = VtraceFirst(
                xis_f=torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                next_values_f=torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                n_rewards_f=torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                vs_f=torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device),
                qs_f=torch.zeros((batch_size, 2, minibatch.policies.shape[2]), dtype=torch.float32, device=self.device)
            )

            chunk_num = (max_game_size + fixed_game_size - 1) // fixed_game_size


            self.optimizer.zero_grad()
            for i in reversed(range(chunk_num)):
                size = min(max_game_size-i*fixed_game_size, fixed_game_size)
                start = i*fixed_game_size
                end = start + size
                policy_loss_i, value_loss_i, loss_i, vtracefirst = self.get_loss(minibatch, vtracefirst, start, end)
                
                policy_loss_i = policy_loss_i/accumration
                value_loss_i = value_loss_i/accumration
                loss_i = loss_i/accumration

                with torch.no_grad():
                    policy_loss += policy_loss_i.item()
                    value_loss += value_loss_i.item()
                    loss += loss_i.item()

                loss_i.backward()

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.c_clip_grad)
        self.optimizer.step()

        with torch.no_grad():
            self.target_network.load_state_dict(
                self.get_EMA_state_dict(self.network,self.target_network, self.gamma_ave)
            )
        
            # 6. Update Step (pi_reg の更新)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.reg_update_interval == 0:
            print("Update Regularization Policy (pi_reg) ...")
            self.prev_reg_network.load_state_dict(self.reg_network.state_dict())
            self.reg_network.load_state_dict(self.target_network.state_dict())
        
        torch.cuda.empty_cache()
        
        self.add_loss_data(loss_print_path, loss, policy_loss, value_loss)
        gc.collect()

    def add_loss_data(self, path:str, loss, p, v):
        if(self.file_inited == False):
            self.init_loss_file(path)
            
        with open(f"{path}/loss.csv", "a") as f:
            f.write(f"{loss},{p},{v}\n")
            f.close()
    
    def init_loss_file(self, path:str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/loss.csv", "w") as f:
            f.write("loss,policy_loss,value_loss\n")
            f.close()
            
        self.file_inited = True
    
    
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
        omega_action = omega.gather(2, actions.unsqueeze(2)).squeeze(2)

        penalty = self.eta * omega_action

        play1 = torch.ones(penalty.shape, dtype=torch.int8, device=self.device)
        play1[players == 0] = -1
        play2 = -1*play1

        pe1 = penalty * play1
        pe2 = penalty * play2
        
        transformed_rewards = rewards.clone()
        transformed_rewards[:,:,0] += pe1
        transformed_rewards[:,:,1] += pe2
        
        return transformed_rewards
    
    def make_reglogit_advantage(self,qs, policy, logits, non_legals, players, masks, i, c_clip):
        ps = (players == i)*masks

        qs = qs[ps,i,:]
        policy = policy[ps,:]
        logits = logits[ps,:]
        legals_i = (non_legals == False)[ps,:]

        legal_logits = torch.where((legals_i == 1),logits, 0)
        legal_count = legals_i.sum(dim=1)

        advantage = (qs - (qs*policy.detach()).sum(dim=1).unsqueeze(1))
        advantage = clip(advantage, c_clip)
            
        # Policy Loss
        reg_logits = logits - (legal_logits.sum(dim=1)/legal_count).unsqueeze(1)

        return reg_logits, advantage, legals_i, ps


    def policy_update(
            self,
            qs, policy, logits, players, masks,
            c_clip_neurd, non_legals: torch.Tensor, beta
        ):
        sum_log_q = 0
        for i in range(2):

            reg_logits, adv, legals_i, ps = self.make_reglogit_advantage(qs, policy, logits, non_legals, players, masks, i, c_clip_neurd)

            with torch.no_grad():
                can_increase = reg_logits < beta
                can_decrease = reg_logits > -beta

                force_positive = torch.maximum(adv, torch.tensor(0.0, device=self.device))
                force_negative = torch.minimum(adv, torch.tensor(0.0, device=self.device))

                cliped_force = can_increase*force_positive + can_decrease*force_negative

            loss = reg_logits * cliped_force

            legal_loss = torch.where((legals_i == 1),loss, 0)
            
            p_effective = ps.sum(dim=1)
            batch_size = p_effective.size(0)
            segment_ids = torch.repeat_interleave(torch.arange(batch_size, device=self.device), p_effective)
            
            log_q = torch.zeros(batch_size, device=self.device)
            log_q = log_q.index_add(0, segment_ids, legal_loss.sum(dim=1))
            
            p_effective[p_effective == 0] = 1
            log_q = log_q / p_effective
            
            sum_log_q += log_q.sum()



        return sum_log_q