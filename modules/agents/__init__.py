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
        self.demands = np.zeros(self.cluster_size)
        self.assignments = np.zeros(self.cluster_size)
        self.demand_history = list()

    def run_agent(self, iteration, assignments):
        raise NotImplementedError

    def get_weight(self):
        return self.weight

    def set_extra(self, extra):
        return

    def get_extra(self):
        return None
    
    def stop(self, path):
        self.dist_app.stop(path)
        self.policy.stop(path, self.agent_id)
