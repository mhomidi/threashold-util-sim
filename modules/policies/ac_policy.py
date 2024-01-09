import os
import sys

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
        self.critic_layer1_norm = nn.LayerNorm(h1_size)
        self.critic_layer2 = nn.Linear(h1_size, h2_size)
        self.critic_layer2_norm = nn.LayerNorm(h2_size)
        self.critic_layer3 = nn.Linear(h2_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.critic_layer1_norm(torch.relu(self.critic_layer1(x)))
        x = self.critic_layer2_norm(torch.relu(self.critic_layer2(x)))
        state_value = self.critic_layer3(x)
        return state_value


class Actor(nn.Module):
    def __init__(self, input_size, output_size, h1_size, lr, std_max, net_type):
        super().__init__()
        a_l1_size = input_size
        self.actor_layer1 = nn.Linear(a_l1_size, h1_size)
        # all norm layer added for avoiding divergence
        self.actor_layer1_norm = nn.LayerNorm(h1_size)
        self.net_type = net_type
        self.output_size = output_size
        if net_type == 'normal':
            self.actor_layer2_mean = nn.Linear(h1_size, 1)
        elif net_type == 'softmax':
            self.actor_layer2 = nn.Linear(h1_size, h1_size)
            self.actor_layer2_norm = nn.LayerNorm(h1_size)
            self.out_layer = nn.Linear(h1_size, output_size)
        else:
            sys.exit('Unrecognized net type')
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.std_max = std_max

    def forward(self, x):
        x = self.actor_layer1_norm(torch.relu(self.actor_layer1(x)))
        if self.net_type == 'normal':
            mean = torch.sigmoid(self.actor_layer2_mean(x))
            dist = Normal(loc=mean, scale=self.std_max)
            u = dist.sample()
            log_prob = dist.log_prob(u)
        elif self.net_type == 'softmax':
            x = self.actor_layer2_norm(torch.relu(self.actor_layer2(x)))
            prob = torch.softmax(self.out_layer(x), dim=-1)
            u = np.random.choice([i/self.output_size for i in range(self.output_size)], p=prob.detach().numpy())
            log_prob = torch.log(prob[int(u * self.output_size)])
        else:
            sys.exit('Unrecognized')
        return u, log_prob

    def get_mean_std(self, x):
        x = torch.relu(self.actor_layer1(x))
        if self.net_type == 'normal':
            mean = torch.sigmoid(self.actor_layer2_mean(x))
            std = self.std_max
            return mean, std
        elif self.net_type == 'softmax':
            x = torch.relu(self.actor_layer2(x))
            prob = torch.softmax(self.out_layer(x), dim=-1)
            return prob
        else:
            return 0


class ACPolicy(Policy):
    def __init__(self, a_input_size, c_input_size, a_h1_size, c_h1_size, c_h2_size,
                 a_lr, c_lr, df, std_max, num_clusters, mini_batch_size=1, threshold_steps=10, actor_net_type='softmax'):
        super().__init__(num_clusters)
        self.actor = Actor(a_input_size, threshold_steps, a_h1_size, a_lr, std_max, net_type=actor_net_type)
        self.critic = Critic(c_input_size, c_h1_size, c_h2_size, c_lr)
        self.log_prob = None
        self.discount_factor = df
        self.state_value = torch.tensor([0.0], requires_grad=True)
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.iteration = 0
        self.mini_batch_size = mini_batch_size
        self.thr_history = []
        self.reward_history = []
        self.threshold = None

    def compute_returns(self, next_state_value):
        r = next_state_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            r = self.rewards[step] + self.discount_factor * r
            returns.insert(0, r)
        return returns

    def get_new_threshold(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        self.threshold, self.log_prob = self.actor(state_tensor)
        return self.threshold.item()

    def get_demands(self, state):
        [normalized_q_lengths, tokens] = state
        state_array = np.concatenate((normalized_q_lengths, tokens))
        threshold = self.get_new_threshold(state_array)
        self.thr_history.append(threshold)
        demands = normalized_q_lengths.copy()
        demands[demands < threshold] = 0
        # print(demands)
        return demands

    def printable_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        return self.actor.get_mean_std(state_tensor)

    def update_policy(self, old_state, action, reward, next_state):
        self.values.append(self.state_value)
        self.rewards.append(reward)
        self.reward_history.append(reward)
        self.log_probs.append(self.log_prob)

        [next_normalized_q_lengths, next_tokens] = next_state
        next_state_array = np.concatenate((next_normalized_q_lengths, next_tokens))
        next_state_tensor = torch.tensor(next_state_array, dtype=torch.float32)
        self.iteration += 1

        if self.iteration == self.mini_batch_size:
            next_state_value = self.critic(next_state_tensor)
            returns = self.compute_returns(next_state_value)
            returns = torch.cat(returns).detach()
            values = torch.cat(self.values)
            advantage = returns - values
            if self.mini_batch_size == 1:
                log_probs = self.log_probs[0]
            else:
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

    def stop(self, path, agent_id):
        agent_path = path + '/agent_' + str(agent_id)
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        np.savetxt(agent_path + '/thresholds.csv',
                   self.thr_history, fmt='%.2f', delimiter=',')
        np.savetxt(agent_path + '/rewards.csv',
                   self.reward_history, fmt='%.2f', delimiter=',')
