from modules.scheduler import Scheduler
import numpy as np


class RoundRobinScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_clusters):
        super().__init__(agent_weights, num_agents, num_clusters)
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        round_turn_starter = 0
        for cluster_idx in range(self.num_clusters):
            self.assignments[round_turn_starter, cluster_idx] = 1
            round_turn_starter += 1
            round_turn_starter %= self.num_agents

    def run_scheduler(self, iteration, demands):
        self.assignments = np.roll(self.assignments, 1, axis=1)
        return self.assignments, None
