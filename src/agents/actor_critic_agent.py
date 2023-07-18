
import torch
import torch.nn as nn
from src.agents import Agent


class ActorCriticAgent(nn.Module, Agent):

    def __init__(self, budget, state_dim, action_dim):
        Agent.__init__(self, budget)
        nn.Module.__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        return pi, v