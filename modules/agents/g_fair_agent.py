from modules.agents import Agent
import numpy as np

from modules.policies.g_fair_policy import GFairPolicy


class GFairAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: GFairPolicy
        # self.speed_up = np.sort(np.random.uniform(0., 1., self.cluster_size))

    def set_extra(self, extra):
        return

    # def get_speed_up(self) -> np.ndarray:
    #     return self.speed_up

    def run_agent(self, iteration, assignments):
        self.dist_app.update_dist_app(iteration, assignments)
        new_state = self.dist_app.get_state()
        self.demands = self.policy.get_demands(new_state)
        return self.demands
