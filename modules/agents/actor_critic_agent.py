
import torch
import torch.nn as nn
from modules.agents import Agent
from config import config
import torch.optim as optim
import numpy as np
from utils.distribution import UniformMeanGenerator
from utils import constant


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1 + config.CLUSTERS_NUM, 64) # +1 for budget
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, config.THRESHOLDS_NUM)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        return pi, v
        

class ActorCriticAgent(Agent):

    def __init__(self, budget: int, 
                 u_gen_type=constant.U_GEN_MARKOV,
                 mean_u_gen=UniformMeanGenerator(),
                 application=None
                 ) -> None:
        super(ActorCriticAgent, self).__init__(budget=budget, u_gen_type=u_gen_type, mean_u_gen=mean_u_gen, application=application)
        self.prev_budget = budget
        self.model = Model()
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.utils = np.random.rand(config.CLUSTERS_NUM).tolist()
        self.prev_utils = self.utils


    def train(self):
        self.round_util = self.get_round_utility()
        next_state = [self.budget] + self.utils

        _, next_val = self.model(torch.tensor(next_state, dtype=torch.float32))
        err = self.round_util + config.DISCOUNT_FACTOR * next_val - self.val

        actor_loss = -torch.log(self.probs[self.u_thr_index]) * err
        critic_loss = torch.square(err)
        loss = actor_loss + critic_loss

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_u_thr(self):
        input_data = [self.budget] + self.utils
        self.probs, self.val = self.model(torch.tensor(input_data, dtype=torch.float32))
        self.u_thr_index = np.random.choice(np.arange(len(self.probs)), p=self.probs.detach().numpy())
        return self.u_thr_index / config.THRESHOLDS_NUM
    
    def set_budget(self, budget: int) -> None:
        self.prev_budget = self.budget
        return super().set_budget(budget)
    
    def update_utils(self):
        self.prev_utils = self.utils
        super().update_utils()

    def get_round_utility(self):
        util = 0.0
        for c_id, agent_id in enumerate(self.assignment):
            if agent_id == self.id:
                util += self.prev_utils[c_id]
        return util
