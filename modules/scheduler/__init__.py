import numpy as np
from modules.agents import *


class Scheduler:
    def __init__(self, agent_weights, num_agents, num_clusters):
        self.agent_weights = agent_weights
        self.num_agents = num_agents
        self.num_clusters = num_clusters
        self.assignments = np.zeros((self.num_agents, self.num_clusters))

    def run_scheduler(self, iteration, demands):
        raise NotImplementedError

    def update_scheduler(self, data):
        return
