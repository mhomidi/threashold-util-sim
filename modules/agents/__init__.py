import numpy as np
from modules.applications.dist_app import DistributedApplication
from modules.policies import Policy


class Agent:
    def __init__(self, agent_id, weight, distributed_app, policy: Policy):
        self.cluster_size = distributed_app.get_cluster_size()
        self.agent_id = agent_id
        self.weight = weight
        self.dist_app = distributed_app
        self.policy = policy
        self.demand = np.zeros(self.cluster_size)
        self.assignments = np.zeros(self.cluster_size)
        self.demand_history = list()

    def run_agent(self, iteration, assignments):
        self.dist_app.update_dist_app(iteration, assignments)
        new_state = self.dist_app.get_customized_state()
        self.demand = self.policy.get_demands(new_state)
        return self.demand

    def get_weight(self):
        return self.weight

    def set_extra(self, extra):
        return

    def get_extra(self):
        return None
    
    def stop(self, path):
        self.dist_app.stop(path)
        self.policy.stop(path, self.agent_id)
