from modules.scheduler import Scheduler
import numpy as np


class GFairScheduler(Scheduler):

    class StrideScheduler:
        def __init__(self, weights):
            self.num_agents = len(weights)
            self.weights = weights
            self.passes = np.zeros(self.num_agents)
            self.strides = 100000 / self.weights

        def schedule(self, demands):
            agents_w_demands = np.arange(0, self.num_agents)[demands > 0]
            if len(agents_w_demands) > 0:
                active_passes = self.passes[agents_w_demands]
                agents_w_demands = agents_w_demands[active_passes == active_passes.min()]
                agent_id = np.random.choice(agents_w_demands)
                self.passes[agent_id] += self.strides[agent_id]
                return agent_id
            else:
                return np.random.choice(np.arange(0, self.num_agents))

    def __init__(self, agent_weights, num_agents, num_nodes):
        super().__init__(agent_weights, num_agents, num_nodes)
        # self.speed_ups = np.zeros((self.num_agents, self.num_nodes))
        self.stride_schedulers = [self.StrideScheduler(
            self.agent_weights) for _ in range(self.num_nodes)]

    def run_scheduler(self, iteration, demands):
        self.assignments = np.zeros((self.num_agents, self.num_nodes))
        for i in range(self.num_nodes):
            agent_id = self.stride_schedulers[i].schedule(demands[:, i])
            if agent_id >= 0:
                self.assignments[(agent_id, i)] = 1

        return self.assignments, None
