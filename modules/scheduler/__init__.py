import numpy as np
from modules.agents import *


class Scheduler:
    def __init__(self, agent_weights, num_agents, num_nodes):
        self.agent_weights = agent_weights
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.assignments = np.zeros((self.num_agents, self.num_nodes))

    def run_scheduler(self, iteration, demands):
        raise NotImplementedError

    def update_scheduler(self, data):
        return

    def get_alloc(self, x):
        x = x / np.sum(x, axis=0)
        allocation = np.zeros((self.num_agents, self.num_nodes))
        for i in range(self.num_nodes):
            index = np.random.choice(range(0, self.num_agents), p=x[:, i])
            allocation[index, i] = 1
        return allocation
