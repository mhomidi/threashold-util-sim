import numpy as np
import config
from modules.agents import *


class Scheduler:
    def __init__(self, agents):
        self.agents = agents
        self.num_agents = config.get(config.AGENT_NUM)
        self.num_clusters = config.get(config.CLUSTER_NUM)
        self.assignments = np.zeros((self.num_agents, self.num_clusters))

    def run_scheduler(self, iteration, demands):
        raise NotImplementedError


class MostTokensFirstScheduler(Scheduler):
    def __init__(self, agents, token_coefficient=100):
        super().__init__(agents)
        self.tokens = np.zeros(self.num_agents)
        for i in range(0, self.num_agents):
            self.tokens[i] = self.agents[i].get_weight() * token_coefficient

    def run_scheduler(self, iteration, demands: np.ndarray):
        # demands[i] = np.array([0, 3, 1, 2, 0, 0, 0, 0, 0])
        # TODO: bring the logic here! return new assignments and tokens once calculated
        assignment = [-1 for _ in range(self.num_clusters)]
        gathered_token = 0
        for _ in range(self.num_clusters):
            max_budget_agent_index = self.find_max_budget_agent_index()
            if max_budget_agent_index == -1:
                self.assgin_randomly_available_clusters(assignment)
                break
            cluster_idx = np.argmax(demands[max_budget_agent_index])
            assignment[cluster_idx] = max_budget_agent_index
            self.tokens[max_budget_agent_index] -= 1
            gathered_token += 1
            demands[:, cluster_idx] = 0
        self.create_one_hot_assignments(assignment)
        self.redistribute_tokens(gathered_token)
        return self.assignments, self.tokens

    def create_one_hot_assignments(self, assignments: list):
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        for c, a in enumerate(assignments):
            self.assignments[a, c] = 1

    def find_max_budget_agent_index(self, demands: np.ndarray) -> int:
        max_budget_agent_indices = np.argsort(self.tokens).tolist()[::-1]
        for i in max_budget_agent_indices:
            if demands[i].sum() >= 0 and self.tokens[i] >= 1.:
                return i
        return -1

    def assgin_randomly_available_clusters(self, assignment: list) -> int:
        weights = np.array([agent.get_weight() for agent in self.agents])
        p = weights / weights.sum()
        for i in assignment:
            if i == -1:
                r = np.random.choice([i for i in range(self.num_agents)], p=p)
                assignment[i] = r

    def redistribute_tokens(self, gathered_tokens: int):
        weights = np.array([agent.get_weight() for agent in self.agents])
        weights /= weights.sum()
        self.tokens += weights * gathered_tokens


class GandivaFairScheduler(Scheduler):

    class StrideScheduler:
        def __init__(self, weights):
            self.num_agents = len(weights)
            self.weights = weights
            self.passes = np.zeros(self.num_agents)
            self.strides = 100000 / self.weights

        def schedule(self, demands):
            agents_w_demands = np.arange(0, self.num_agents)[demands > 0]
            agent_id = -1
            if len(agents_w_demands) > 0:
                active_passes = self.passes[agents_w_demands]
                agent_id = agents_w_demands[np.argmin(active_passes)]
                self.passes[agent_id] += self.strides[agent_id]
            return agent_id

    def __init__(self, agents):
        super().__init__(agents)
        self.speed_ups = np.zeros((self.num_agents, self.num_clusters))

        self.weights = np.zeros((self.num_agents, self.num_clusters))
        for agent in agents:
            agent: GandivaFairAgent
            self.speed_ups[agent.agent_id] = agent.get_speed_up()
            self.weights[agent.agent_id] = agent.get_weight()
        self.trade_resources()
        self.stride_schedulers = [self.StrideScheduler(
            self.weights[:, i].tolist()) for i in range(self.num_clusters)]

    def run_scheduler(self, iteration, demands):
        # demands[i] = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0])
        # TODO: bring the logic here! return new assignments
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        for i in range(self.num_clusters):
            agent_id = self.stride_schedulers[i].schedule(demands[:, i])
            if agent_id >= 0:
                self.assignments[agent_id][i] = 1.

        return self.assignments

    def trade_resources(self):
        for low in range(config.get(config.CLUSTER_NUM)):
            high = config.get(config.CLUSTER_NUM) - 1
            while high > low:
                while True:
                    f_max_index, s_max_index, min_index = self.get_available_indices(
                        self.speed_ups.copy(), low, high)
                    if f_max_index == min_index or f_max_index == None or min_index == None:
                        break
                    speed_up = self.speed_ups[s_max_index][high] / \
                        self.speed_ups[s_max_index][low]
                    self.trade(f_max_index, min_index,
                               speed_up, low=low, high=high)
                high -= 1

    def trade(self, f_max_index: int, min_index: int, speed_up: float, low: int = 0, high: int = config.get(config.CLUSTER_NUM) - 1):
        low_f_max_weight = self.weights[f_max_index][low]
        high_min_weight = self.weights[min_index][high]

        if low_f_max_weight / speed_up > high_min_weight:
            self.weights[f_max_index][low] -= high_min_weight * speed_up
            self.weights[min_index][low] += high_min_weight * speed_up
            self.weights[f_max_index][high] += high_min_weight
            self.weights[min_index][high] = 0.0
        else:
            self.weights[f_max_index][low] = 0.0
            self.weights[min_index][low] += low_f_max_weight
            self.weights[f_max_index][high] += low_f_max_weight / speed_up
            self.weights[min_index][high] -= low_f_max_weight / speed_up

    def get_available_indices(self, throughput: np.ndarray, low_util_index: int, high_util_index: int) -> tuple:
        f_max_index = None
        s_max_index = None
        min_index = None
        high_util_throughput = throughput[:, high_util_index]
        low_util_throughput = throughput[:, low_util_index]
        high_util_throughput = high_util_throughput / low_util_throughput
        low_util_throughput = np.ones(low_util_throughput.shape)
        sorted_args = np.argsort(high_util_throughput).tolist()
        for arg in sorted_args:
            if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
                min_index = arg
                break
        for idx, arg in enumerate(sorted_args[::-1]):
            if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
                f_max_index = arg
                if idx < len(sorted_args) - 1:
                    s_max_index = sorted_args[::-1][idx + 1]
                break
        return f_max_index, s_max_index, min_index
