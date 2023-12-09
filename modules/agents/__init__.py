import numpy as np


class Agent:
    def __init__(self, agent_id, weight, distributed_app, policy):
        self.cluster_size = distributed_app.get_cluster_size()
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

    def set_extra(self, extra):
        raise NotImplementedError

    def stop(self, path):
        self.dist_app.stop(path)
