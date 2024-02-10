from modules.scheduler import Scheduler
import numpy as np


class WeightedRoundRobinScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_clusters):
        super().__init__(agent_weights, num_agents, num_clusters)
        self.index = 0
        self.turns = []
        for i in range(num_agents):
            for _ in range(agent_weights[i]):
                self.turns.append(i)

    def run_scheduler(self, iteration, demands):
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        for c in range(self.num_clusters):
            agent = self.turns[self.index]
            cluster = np.argmax(demands[agent])
            self.assignments[(agent, cluster)] = 1
            demands[:, cluster] = -1
            self.index = (self.index + 1) % self.num_clusters
        return self.assignments, None
