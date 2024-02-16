from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np

FTF_ALPHA = 0.9


class ThroughputFairScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes, departure_rates, arrival_rates):
        super().__init__(agent_weights, num_agents, num_nodes)
        weights = self.agent_weights / self.agent_weights.sum()
        shares = np.ones((self.num_agents, self.num_nodes)) * weights.reshape((self.num_agents, 1))
        self.departure_rates = departure_rates
        self.arrival_rates = arrival_rates.reshape((self.num_agents, 1))
        exclusive_departure_rates = self.departure_rates.sum(axis=1, keepdims=True) * shares
        self.exclusive_throughput = np.minimum(self.arrival_rates, exclusive_departure_rates)
        self.shared_throughput = np.zeros((self.num_agents, 1))


    def run_scheduler(self, iteration, demands):
        ones_c = np.ones(self.num_nodes)
        ones_a = np.ones((self.num_agents, 1))
        ones_ac = np.ones((self.num_agents, self.num_nodes))

        rho = cp.Variable(1)
        x = cp.Variable((self.num_agents, self.num_nodes), nonneg=True)
        shared_departures = cp.sum(cp.minimum(cp.multiply(x, self.departure_rates), demands), axis=1, keepdims=True)
        shared_throughput = FTF_ALPHA * self.shared_throughput + (1 - FTF_ALPHA) * shared_departures

        constraints = [
            cp.sum(x, axis=0) <= ones_c,
            rho * ones_a <= shared_throughput / self.exclusive_throughput
        ]

        objective = cp.Maximize(rho)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        if problem.status != cp.OPTIMAL:
            raise Exception(problem.status)

        allocation = self.get_alloc(x.value)
        # after alloc, recalculate shared_departures based on alloc
        shared_departures = np.minimum(allocation * self.departure_rates, demands)
        self.shared_throughput = FTF_ALPHA * self.shared_throughput + (1 - FTF_ALPHA) * shared_departures
        return allocation, None
