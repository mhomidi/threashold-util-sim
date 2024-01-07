from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np


class FinishTimeFairnessScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_clusters, departure_rates, arrival_rates):
        super().__init__(agent_weights, num_agents, num_clusters)
        weights = self.agent_weights / self.agent_weights.sum()
        shares = np.ones((self.num_agents, self.num_clusters)) * weights.reshape((self.num_agents, 1))
        self.departure_rates = departure_rates
        self.arrival_rates = arrival_rates
        weights = self.agent_weights / self.agent_weights.sum()
        exclusive_departure_rates = self.departure_rates.sum(axis=1) * weights
        exclusive_utilization = self.arrival_rates / exclusive_departure_rates
        assert np.max(exclusive_utilization) < 1
        self.exclusive_q_lengths = exclusive_utilization / (1 - exclusive_utilization)
        self.shared_queue_lengths = 0

    # TODO: get rid of set_more_data
    # TODO: for now, do not call this function anywhere
    def update_scheduler(self, data):
        self.departure_rates, self.arrival_rates = data
        weights = self.agent_weights / self.agent_weights.sum()
        exclusive_departure_rates = self.departure_rates.sum(axis=1) * weights
        exclusive_utilization = self.arrival_rates / exclusive_departure_rates
        assert np.max(exclusive_utilization) < 1
        self.exclusive_q_lengths = exclusive_utilization / (1 - exclusive_utilization)

    def run_scheduler(self, iteration, demands) -> list:
        ones_c = np.ones(self.num_clusters)
        ones_ac = np.ones((self.num_agents, self.num_clusters))

        rho = cp.Variable(1)
        x = cp.Variable((self.num_agents, self.num_clusters), boolean=True)
        shared_departure_rates = cp.multiply(x, self.departure_rates)
        # TODO: Experiment with this!
        shared_queue_length = 0.9 * self.shared_queue_lengths + 0.1 * (demands - shared_departure_rates)

        exc_q_lengths = ones_ac * self.exclusive_q_lengths
        constraints = [cp.sum(x, axis=0) == ones_c, rho * ones_ac >= shared_queue_length / exc_q_lengths]
        objective = cp.Minimize(rho)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status != cp.OPTIMAL:
            raise Exception("aborted!")
        self.shared_queue_lengths = shared_queue_length.value
        return x.value, None
