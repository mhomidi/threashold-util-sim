from modules.scheduler import Scheduler
import numpy as np

class StrideScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes):
        super().__init__(agent_weights, num_agents, num_nodes)
        self.agent_weights = self.agent_weights / self.agent_weights.sum()

    def run_scheduler(self, iteration, demands):
        self.assignments = np.zeros((self.num_agents, self.num_nodes))
        random_agents = np.random.choice(range(self.num_agents), size=self.num_clusters, p=self.agent_weights)
        for agent in random_agents:
            preferred_cluster = np.argmax(demands[agent])
            self.assignments[(agent, preferred_cluster)] = 1
            demands[:, preferred_cluster] = -1
        return self.assignments, None