import os
from modules.agents import Agent
import numpy as np
from modules.policies.ac_policy import ACPolicy


class ACAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy, tokens):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: ACPolicy
        self.tokens = tokens
        self.new_tokens = tokens
        self.demand = self.policy.get_demands([self.dist_app.get_normalized_state(), self.tokens])

    def set_extra(self, tokens):
        self.new_tokens = tokens

    def run_agent(self, iteration, assignments):
        old_state = [self.dist_app.get_normalized_state(), self.tokens]
        self.dist_app.update_dist_app(iteration, assignments)
        self.tokens = self.new_tokens
        reward = self.dist_app.get_utility()
        new_state = [self.dist_app.get_normalized_state(), self.tokens]
        self.policy.update_policy(old_state, self.demands, reward, new_state)
        self.demands = self.policy.get_demands(new_state)
        self.demand_history.append(self.demands)
        return self.demands
    
    def get_extra(self):
        return self.policy.get_threshold()

    def stop(self, path):
        super().stop(path)
        agent_path = path + '/agent_' + str(self.agent_id)
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        np.savetxt(agent_path + '/demands.csv',
                   np.array(self.demand_history), fmt='%.2f', delimiter=',')
