
from src.Agent.IS_MCTS.network import IsMctsNetwork
from src.Agent.IS_MCTS.replay_buffer import ReplayBuffer

import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsMctsLearner:
    def __init__(
            self, device: torch.device,
            in_channels: int = 64, mid_channels: int = 82,
            lr = 0.01
        ):
        self.network = IsMctsNetwork(in_channels, mid_channels, device = device)

        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)

        self.losses = []
        

    def learn(self, replay_buffer: ReplayBuffer, batch_size:int = 32, loss_print_path: str = "model_loss/is_mcts"):
        batch = replay_buffer.sample(batch_size)

        boards = batch.boards.to(self.device)
        rewards = batch.rewards.to(self.device)

        values = self.network(boards)

        loss = F.mse_loss(rewards, values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss)

        plt.plot(self.losses, label="合計loss")
        plt.xlabel("epoc")
        plt.legend()
        plt.grid()
        plt.savefig(f"{loss_print_path}/loss.png", format="png")
        plt.cla()
        plt.clf()
        plt.close()



        
        
