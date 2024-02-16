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
