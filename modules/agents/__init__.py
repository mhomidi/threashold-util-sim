import config
import numpy as np

from modules.policies import ACPolicy, GandivaFairPolicy


class Agent:
    def __init__(self, agent_id, weight, distributed_app, policy):
        self.cluster_size = config.get(config.CLUSTER_NUM)
        self.agent_id = agent_id
        self.weight = weight
        self.dist_app = distributed_app
        self.policy = policy
        self.demands = np.zeros(self.cluster_size)
        self.assignments = np.zeros(self.cluster_size)

    def run_agent(self, iteration, assignments):
        raise NotImplementedError

    def get_weight(self):
        return self.weight


class ACAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: ACPolicy
        self.tokens = np.zeros(config.get(config.AGENT_NUM))
        self.new_tokens = np.zeros(config.get(config.AGENT_NUM))

    def set_tokens(self, tokens):
        self.new_tokens = tokens

    def run_agent(self, iteration, assignments):
        old_state = np.array([self.dist_app.get_curr_state(), self.tokens])
        self.dist_app.update_dist_app(iteration, assignments)
        self.tokens = self.new_tokens
        reward = self.dist_app.get_utility()
        new_state = np.array([self.dist_app.get_curr_state(), self.tokens])
        self.policy.update_policy(old_state, self.demands, reward, new_state)
        self.demands = self.policy.get_demands(new_state)
        return self.demands


class GandivaFairAgent(Agent):
    def __init__(self, agent_id, weight, distributed_app, policy):
        super().__init__(agent_id, weight, distributed_app, policy)
        self.policy: GandivaFairPolicy
        self.speed_up = np.sort(np.random.uniform(0., 1., self.cluster_size))

    def get_speed_up(self) -> np.ndarray:
        return self.speed_up

    def run_agent(self, iteration, assignments):
        self.dist_app.update_dist_app(iteration, assignments)
        new_state = self.dist_app.get_curr_state()
        self.demands = self.policy.get_demands(new_state)
        return self.demands
