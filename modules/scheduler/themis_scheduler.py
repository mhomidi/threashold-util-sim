from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np
import pandas as pd


FTF_ALPHA = 0.9


class QLengthFairScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_nodes, departure_rates, arrival_rates):
        super().__init__(agent_weights, num_agents, num_nodes)
        weights = self.agent_weights / self.agent_weights.sum()
        shares = np.ones((self.num_agents, self.num_nodes)) * weights.reshape((self.num_agents, 1))
        self.departure_rates = departure_rates
        self.arrival_rates = arrival_rates
        exclusive_departure_rates = self.departure_rates.sum(axis=1, keepdims=True) * shares
        exclusive_utilization = self.arrival_rates / exclusive_departure_rates
        assert np.max(exclusive_utilization) < 1
        self.exclusive_q_lengths = exclusive_utilization / (1 - exclusive_utilization)
        self.shared_queue_lengths = np.zeros((self.num_agents, self.num_nodes))

    # TODO: for now, do not call this function anywhere
    # def update_scheduler(self, data):
    #     self.departure_rates, self.arrival_rates = data
    #     weights = self.agent_weights / self.agent_weights.sum()
    #     exclusive_departure_rates = self.departure_rates.sum(axis=1) * weights
    #     exclusive_utilization = self.arrival_rates / exclusive_departure_rates
    #     assert np.max(exclusive_utilization) < 1
    #     self.exclusive_q_lengths = exclusive_utilization / (1 - exclusive_utilization)

    def run_scheduler(self, iteration, demands):
        ones_c = np.ones(self.num_nodes)
        ones_ac = np.ones((self.num_agents, self.num_nodes))

        rho = cp.Variable(1)
        x = cp.Variable((self.num_agents, self.num_nodes), boolean=True)
        shared_departure_rates = cp.multiply(x, self.departure_rates)
        shared_queue_length = FTF_ALPHA * self.shared_queue_lengths + (1 - FTF_ALPHA) * (demands - shared_departure_rates)
        # shared_queue_length = demands - cp.multiply(x, self.departure_rates)

        exc_q_lengths = ones_ac * self.exclusive_q_lengths
        constraints = [
            cp.sum(x, axis=0) <= ones_c,
            shared_departure_rates <= demands,
            rho * ones_ac >= shared_queue_length / exc_q_lengths
        ]
        objective = cp.Minimize(rho)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status != cp.OPTIMAL:
            raise Exception(problem.status)
        self.shared_queue_lengths = shared_queue_length.value
        allocation = np.array(x.value > 0.5, dtype=np.int8)
        return allocation, None


class EEThroughputFairScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes, departure_rates, arrival_rates):
        super().__init__(agent_weights, num_agents, num_nodes)
        weights = self.agent_weights / self.agent_weights.sum()
        shares = np.ones((self.num_agents, self.num_nodes)) * weights.reshape((self.num_agents, 1))
        self.departure_rates = departure_rates
        self.arrival_rates = arrival_rates.reshape((self.num_agents, 1))
        exclusive_departure_rates = self.departure_rates.sum(axis=1, keepdims=True) * shares
        self.exclusive_throughput = np.minimum(self.arrival_rates, exclusive_departure_rates)
        ###
        self.shared_throughput = np.zeros((self.num_agents, 1))
        ###

    def run_scheduler(self, iteration, demands):
        ones_c = np.ones(self.num_nodes)
        ones_a = np.ones((self.num_agents, 1))
        ones_ac = np.ones((self.num_agents, self.num_nodes))

        rho = cp.Variable(1)
        x = cp.Variable((self.num_agents, self.num_nodes), boolean=True)
        ###
        shared_departures = cp.sum(cp.minimum(cp.multiply(x, self.departure_rates), demands), axis=1, keepdims=True)
        shared_throughput = FTF_ALPHA * self.shared_throughput + (1 - FTF_ALPHA) * shared_departures
        ###
        # shared_throughput = shared_departures

        constraints = [
            cp.sum(x, axis=0) <= ones_c,
            # cp.multiply(x, self.departure_rates) <= demands,
            # y <= demands,
            # y <= cp.multiply(x, self.departure_rates),
            rho * ones_a <= shared_throughput / self.exclusive_throughput
        ]
        objective = cp.Maximize(rho)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        if problem.status != cp.OPTIMAL:
            raise Exception(problem.status)
        ###
        self.shared_throughput = shared_throughput.value
        ###
        allocation = np.array(x.value > 0.5, dtype=np.int8)
        return allocation, None


class ThroughputFairScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes, departure_rates, arrival_rates):
        super().__init__(agent_weights, num_agents, num_nodes)
        weights = self.agent_weights / self.agent_weights.sum()
        shares = np.ones((self.num_agents, self.num_nodes)) * weights.reshape((self.num_agents, 1))
        self.departure_rates = departure_rates
        self.arrival_rates = arrival_rates.reshape((self.num_agents, 1))
        exclusive_departure_rates = self.departure_rates.sum(axis=1, keepdims=True) * shares
        self.exclusive_throughput = np.minimum(self.arrival_rates, exclusive_departure_rates)
        ###
        self.shared_throughput = np.zeros((self.num_agents, 1))
        ###

    def run_scheduler(self, iteration, demands):
        ones_c = np.ones(self.num_nodes)
        ones_a = np.ones((self.num_agents, 1))
        ones_ac = np.ones((self.num_agents, self.num_nodes))

        rho = cp.Variable(1)
        x = cp.Variable((self.num_agents, self.num_nodes), nonneg=True)
        ###
        shared_departures = cp.sum(cp.minimum(cp.multiply(x, self.departure_rates), demands), axis=1, keepdims=True)
        # shared_throughput = FTF_ALPHA * self.shared_throughput + (1 - FTF_ALPHA) * shared_departures
        ###
        # shared_throughput = shared_departures

        constraints = [
            cp.sum(x, axis=0) <= ones_c,
            x <= ones_ac,
            rho * ones_a <= shared_departures / self.exclusive_throughput
        ]
        objective = cp.Maximize(rho)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        if problem.status != cp.OPTIMAL:
            raise Exception(problem.status)
        ###
        # self.shared_throughput = shared_throughput.value
        ###
        # Normalize x per column
        norm_x = x.value / x.value.sum(axis=0)
        # use column as prob for sampling => alloc
        allocation = self.get_alloc(norm_x)
        # after alloc, recalculate shared_departures based on alloc
        shared_departures = np.minimum(allocation * self.departure_rates, demands)
        self.shared_throughput = FTF_ALPHA * self.shared_throughput + (1 - FTF_ALPHA) * shared_departures
        return allocation, None
    
    def get_alloc(self, x):
        x = x / np.sum(x, axis=0)
        allocation = np.zeros((self.num_agents, self.num_nodes))
        for i in range(self.num_nodes):
            index = np.random.choice(range(0, self.num_agents), p=x[:, i])
            allocation[index, i] = 1
        return allocation
        # prob = norm_x.T
        # df_probabilities = pd.DataFrame(data=prob, columns=range(self.num_agents))
        # rng = np.random.default_rng()
        # df_selections = pd.DataFrame(
        # data = rng.multinomial(n=1, pvals=df_probabilities), columns=range(self.num_agents))
        # agents = df_selections.idxmax(axis=1).values
        # alloc = np.zeros((self.num_agents, self.num_nodes))
        # alloc[agents,range(self.num_nodes)] = 1
        # return alloc