from modules.scheduler import Scheduler
import numpy as np


class MTFScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes, token_coefficient=100):
        super().__init__(agent_weights, num_agents, num_nodes)
        self.tokens = np.zeros(self.num_agents)
        for i in range(0, self.num_agents):
            self.tokens[i] = self.agent_weights[i] * token_coefficient
        self.thresholds = np.zeros((num_agents, 1))

    def update_scheduler(self, data):
        self.thresholds = np.array(data).reshape(self.num_agents, 1)
        self.thresholds[self.thresholds is None] = 1.0

    def run_scheduler(self, iteration, demands: np.ndarray):
        # token_demand = demands[demands > ]
        token_demands = self.get_token_demands(demands)
        self.assignments = np.zeros((self.num_agents, self.num_nodes))
        gathered_tokens = 0
        tokens = self.tokens.copy()
        while gathered_tokens < self.num_nodes:
            max_budget_agent_index = np.argmax(tokens)
            if tokens[max_budget_agent_index] < 1:
                break
            preferred_cluster = np.argmax(token_demands[max_budget_agent_index])
            if token_demands[max_budget_agent_index, preferred_cluster] == 0:
                tokens[max_budget_agent_index] = 0
                continue
            self.assignments[(max_budget_agent_index, preferred_cluster)] = 1
            tokens[max_budget_agent_index] -= 1
            self.tokens[max_budget_agent_index] -= 1
            gathered_tokens += 1
            token_demands[:, preferred_cluster] = -1
            demands[:, preferred_cluster] = -1
        
        weights = self.agent_weights / self.agent_weights.sum()
        self.tokens += weights * gathered_tokens
        # self.assign_remaining_e(demands)
        self.assign_remaining_r()
        return self.assignments, self.tokens
    
    def get_token_demands(self, demands):
        token_demands = demands.copy()
        for row in range(demands.shape[0]):
            token_demands[row, demands[row] < self.thresholds[row]] = 0.
        return token_demands

    def assign_remaining_e(self, demands):
        not_assigned_num = self.num_nodes - self.assignments.sum(dtype=np.int32)
        random_agents = np.random.choice(range(self.num_agents), size=not_assigned_num, p=self.agent_weights)
        for agent in random_agents:
            preferred_cluster = np.argmax(demands[agent])
            self.assignments[(agent, preferred_cluster)] = 1
            demands[:, preferred_cluster] = -1

    def assign_remaining_r(self):
        not_assigned = np.arange(self.num_nodes)[self.assignments.sum(axis=0) == 0]
        for i in not_assigned:
            agent = np.random.choice(range(self.num_agents), p=self.agent_weights)
            self.assignments[(agent, i)] = 1