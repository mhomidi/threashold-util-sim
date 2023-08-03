
from modules.policies import Policy
from config import config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ACModel(nn.Module):

    def __init__(self):
        super(ACModel, self).__init__()
        self.fc1 = nn.Linear(1 + config.CLUSTERS_NUM, 64) # +1 for budget
        self.actor_fc2 = nn.Linear(64, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.fc_v = nn.Linear(64, 1)
        self.fc_pi = nn.Linear(64, config.THRESHOLDS_NUM)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.actor_fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)

        x = torch.relu(self.fc1(state))
        x = torch.relu(self.critic_fc2(x))
        v: torch.float32 = self.fc_v(x)
        return pi, v


class ActorCriticPolicy(Policy):
    
    def __init__(self, budget: int, ) -> None:
        super().__init__()
        self.budget = budget
        self.prev_budget = budget
        self.ac_model = ACModel()
        self.learning_rate = 0.001
        self.actor_optimizer = optim.Adam(self.ac_model.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.ac_model.parameters(), lr=self.learning_rate)

    def get_u_thr(self, state_data: list):
        self.probs, self.val = self.ac_model(torch.tensor(state_data, dtype=torch.float32))
        self.u_thr_index = np.random.choice(np.arange(len(self.probs)), p=self.probs.detach().numpy())
        return self.u_thr_index / config.THRESHOLDS_NUM
    

    def train(self, reward: float, new_state_data: list):
        _, next_val = self.ac_model(torch.tensor(new_state_data, dtype=torch.float32))
        err = reward + config.DISCOUNT_FACTOR * next_val - self.val

        actor_loss = -torch.log(self.probs[self.u_thr_index]) * err.detach()
        critic_loss = torch.square(err)
        # loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()