from modules.policies import Policy
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

class Critic(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, lr):
        super().__init__()
        c_l1_size = input_size
        self.critic_layer1 = nn.Linear(c_l1_size, h1_size)
        self.critic_layer2 = nn.Linear(h1_size, h2_size)
        self.critic_layer3 = nn.Linear(h2_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.critic_layer1(x))
        x = torch.relu(self.critic_layer2(x))
        state_value = self.critic_layer3(x)
        return state_value


class Actor(nn.Module):
    def __init__(self, input_size, h1_size, lr, std_max):
        super().__init__()
        a_l1_size = input_size
        self.actor_layer1 = nn.Linear(a_l1_size, h1_size)
        self.actor_layer2_mean = nn.Linear(h1_size, 1)
        self.actor_layer2_std = nn.Linear(h1_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.std_max = std_max

    def forward(self, x):
        x = torch.relu(self.actor_layer1(x))
        mean = self.actor_layer2_mean(x)
        std = self.std_max
        dist = Normal(loc=mean, scale=std)
        u = dist.sample()
        log_prob = dist.log_prob(u)
        return u, log_prob

    def get_mean_std(self, x):
        x = torch.relu(self.actor_layer1(x))
        mean = self.actor_layer2_mean(x)
        std = self.std_max
        return mean, std


class ACPolicy(Policy):
    def __init__(self, a_input_size, c_input_size, a_h1_size, c_h1_size, c_h2_size,
                 a_lr, c_lr, df, std_max, num_clusters, mini_batch_size=1):
        super().__init__(num_clusters)
        self.actor = Actor(a_input_size, a_h1_size, a_lr, std_max)
        self.critic = Critic(c_input_size, c_h1_size, c_h2_size, c_lr)
        self.log_prob = None
        self.discount_factor = df
        self.state_value = torch.tensor([0.0], requires_grad=True)
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.iteration = 0
        self.mini_batch_size = mini_batch_size

    def compute_returns(self, next_state_value):
        r = next_state_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            r = self.rewards[step] + self.discount_factor * r
            returns.insert(0, r)
        return returns

    def get_new_threshold(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        threshold, self.log_prob = self.actor(state_tensor)
        return threshold.item()

    def get_demands(self, state):
        [loads, tokens] = state
        state_array = np.concatenate((loads, tokens))
        threshold = self.get_new_threshold(state_array)
        demands = loads.copy()
        demands[demands < threshold] = 0
        return demands

    def printable_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return self.actor.get_mean_std(state_tensor)

    def update_policy(self, old_state, action, reward, next_state):
        self.values.append(self.state_value)
        self.rewards.append(reward)
        self.log_probs.append(self.log_prob)

        [next_load, next_tokens] = next_state
        next_state_array = np.concatenate((next_load, next_tokens))
        next_state_tensor = torch.tensor(next_state_array, dtype=torch.float32)
        self.iteration += 1

        if self.iteration == self.mini_batch_size:
            next_state_value = self.critic(next_state_tensor)
            returns = self.compute_returns(next_state_value)
            returns = torch.cat(returns).detach()
            values = torch.cat(self.values)
            advantage = returns - values
            log_probs = torch.cat(self.log_probs)
            actor_loss = -(log_probs * advantage.detach()).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            critic_loss = advantage.pow(2).mean()

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            self.rewards = []
            self.values = []
            self.log_probs = []
            self.iteration = 0

        self.state_value = self.critic(next_state_tensor)