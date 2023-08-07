
from modules.policies import Policy
import config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ACModel(nn.Module):

    def __init__(self):
        super(ACModel, self).__init__()
        self.fc1 = nn.Linear(2 * (1 + config.get('cluster_num') + config.get('token_dist_sample')), config.get('l1_in')) # +1 for budget
        self.actor_fc2 = nn.Linear(config.get('l1_in'), config.get('actor_l2_in'))
        self.actor_fc3 = nn.Linear(config.get('actor_l2_in'), config.get('actor_l3_in'))
        self.actor_fc4 = nn.Linear(config.get('actor_l3_in'), config.get('actor_l4_in'))
        self.critic_fc2 = nn.Linear(config.get('l1_in'), config.get('critic_l2_in'))
        self.critic_fc3 = nn.Linear(config.get('critic_l2_in'), config.get('critic_l3_in'))
        self.critic_fc4 = nn.Linear(config.get('critic_l3_in'), config.get('critic_l4_in'))
        self.fc_v = nn.Linear(config.get('critic_l4_in'), 1)
        self.fc_pi = nn.Linear(config.get('actor_l4_in'), config.get('threshold_num'))

    def forward(self, state):
        state = torch.cat((state, torch.square(state)))
        x = torch.relu(self.fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        x = torch.tanh(self.actor_fc3(x))
        x = torch.relu(self.actor_fc4(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)

        x = torch.relu(self.fc1(state))
        x = torch.tanh(self.critic_fc2(x))
        x = torch.tanh(self.critic_fc3(x))
        x = torch.relu(self.critic_fc4(x))
        v: torch.float32 = self.fc_v(x)
        return pi, v


class ActorCriticPolicy(Policy):
    
    def __init__(self, budget: int, ) -> None:
        super().__init__()
        self.budget = budget
        self.prev_budget = budget
        self.ac_model = ACModel()
        self.actor_optimizer = optim.Adam(self.ac_model.parameters(), lr=config.get('actor_lr'))
        self.critic_optimizer = optim.Adam(self.ac_model.parameters(), lr=config.get('critic_lr'))

    def get_u_thr(self, state_data: list):
        self.probs, self.val = self.ac_model(torch.tensor(state_data, dtype=torch.float32))
        self.u_thr_index = np.random.choice(np.arange(len(self.probs)), p=self.probs.detach().numpy())
        return self.u_thr_index / config.get('threshold_num')
    

    def train(self, reward: float, new_state_data: list):
        _, next_val = self.ac_model(torch.tensor(new_state_data, dtype=torch.float32))
        err = reward + config.get('discount_factor') * next_val - self.val

        actor_loss = -torch.log(self.probs[self.u_thr_index]) * err.detach()
        critic_loss = torch.square(err)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()