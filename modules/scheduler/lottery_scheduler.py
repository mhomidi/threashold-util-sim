from modules.scheduler import Scheduler
import numpy as np


class LotteryScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_nodes):
        super().__init__(agent_weights, num_agents, num_nodes)
        self.index = 0
        self.turns = []
        for i in range(num_agents):
            for _ in range(int(agent_weights[i])):
                self.turns.append(i)

    def run_scheduler(self, iteration, demands):
        self.assignments = np.zeros((self.num_agents, self.num_nodes))
        for c in range(self.num_nodes):
            agent = self.turns[self.index]
            cluster = np.argmax(demands[agent])
            self.assignments[(agent, cluster)] = 1
            demands[:, cluster] = -1
            self.index = (self.index + 1) % self.num_nodes
        return self.assignments, None

