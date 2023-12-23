from modules.agents import Agent
import numpy as np

from modules.policies.ftf_policy import ThemisPolicy


class ThemisAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: ThemisPolicy
        self.init_demand = self.policy.get_demands(self.dist_app.get_state())

    def run_agent(self, iteration, assignments):
        self.dist_app.update_dist_app(iteration, assignments)
        new_state = self.dist_app.get_state()
        self.demands = self.policy.get_demands(new_state)
        return self.demands

    def get_more_data(self):
        # return avg_throughputs
        return self.dist_app.get_avg_throughput()
