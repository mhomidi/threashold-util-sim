from modules.agents import Agent
import numpy as np
from modules.policies.ac_policy import ACPolicy


class ACAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy, tokens):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: ACPolicy
        self.tokens = tokens
        self.new_tokens = tokens

    def set_extra(self, tokens):
        self.new_tokens = tokens

    def run_agent(self, iteration, assignments):
        old_state = np.array([self.dist_app.get_state(), self.tokens])
        self.dist_app.update_dist_app(iteration, assignments)
        self.tokens = self.new_tokens
        reward = self.dist_app.get_utility()
        new_state = np.array([self.dist_app.get_state(), self.tokens])
        self.policy.update_policy(old_state, self.demands, reward, new_state)
        self.demands = self.policy.get_demands(new_state)
        return self.demands
