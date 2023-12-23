from modules.scheduler import Scheduler
import numpy as np


class MTFScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_clusters, token_coefficient=100):
        super().__init__(agent_weights, num_agents, num_clusters)
        self.tokens = np.zeros(self.num_agents)
        for i in range(0, self.num_agents):
            self.tokens[i] = self.agent_weights[i] * token_coefficient

    def run_scheduler(self, iteration, demands: np.ndarray):
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        gathered_tokens = 0
        tokens = self.tokens.copy()
        while gathered_tokens < self.num_clusters:
            max_budget_agent_index = np.argmax(tokens)
            if tokens[max_budget_agent_index] == 0:
                break
            preferred_cluster = np.argmax(demands[max_budget_agent_index])
            if demands[max_budget_agent_index, preferred_cluster] == 0:
                tokens[max_budget_agent_index] = 0
                continue
            self.assignments[(max_budget_agent_index, preferred_cluster)] = 1
            tokens[max_budget_agent_index] -= 1
            self.tokens[max_budget_agent_index] -= 1
            gathered_tokens += 1
            demands[:, preferred_cluster] = 0

        self.tokens += self.agent_weights * gathered_tokens
        return self.assignments, self.tokens
