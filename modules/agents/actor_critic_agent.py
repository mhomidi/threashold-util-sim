
import torch
import torch.nn as nn
from modules.agents import Agent
from config.config import *
import torch.optim as optim
import numpy as np


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 64) # +1 for budget
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, THRESHOLDS_NUM)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        return pi, v
        

class ActorCriticAgent(Agent):

    def __init__(self, budget: int) -> None:
        super(ActorCriticAgent, self).__init__(budget)
        self.prev_budget = budget
        self.model = Model()
        self.learning_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.utils = np.random.rand(CLUSTERS_NUM).tolist()


    def train(self):
        reward = self.calculate_util(self.assignment)
        print("a{id} gets {reward:.2f} with b={budget}".format(id=self.id+1, reward=reward, budget=self.prev_budget))
        next_state = self.budget

        _, next_val = self.model(torch.tensor([next_state], dtype=torch.float32))
        err = reward + DISCOUNT_FACTOR * next_val - self.val

        actor_loss = -torch.log(self.probs[self.u_thr_index]) * err
        critic_loss = torch.square(err)
        loss = actor_loss + critic_loss

        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_u_thr(self):
        input_data = [self.budget]
        self.probs, self.val = self.model(torch.tensor(input_data, dtype=torch.float32))
        self.u_thr_index = np.random.choice(np.arange(len(self.probs)), p=self.probs.detach().numpy())
        return self.u_thr_index / THRESHOLDS_NUM
    
    def set_budget(self, budget: int) -> None:
        self.prev_budget = self.budget
        return super().set_budget(budget)
    
    def calculate_util(self, assignments: list):
        util = 0.0
        for c_id, agent_id in enumerate(assignments):
            if agent_id == self.id:
                util += self.utils[c_id]
        return util