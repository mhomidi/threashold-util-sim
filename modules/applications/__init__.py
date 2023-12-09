import numpy as np


class Application:

    def __init__(self) -> None:
        self.state_history = list()
        self.state = None

    def update_state(self) -> None:
        raise NotImplementedError

    def get_state(self):
        return self.state

    def stop(self, path):
        raise NotImplementedError


class DistributedApplication:
    def __init__(self, applications):
        self.applications = applications
        self.cluster_size = len(applications)
        self.assignments = np.zeros(self.cluster_size)
        self.utility = 0
        self.utility_history = list()

    def update_dist_app(self, iteration, assignments):
        self.utility_history.append(self.utility)

    def get_utility(self):
        return self.utility

    def get_state(self):
        states = np.zeros(self.cluster_size)
        for i in range(0, self.cluster_size):
            states[i] = self.applications[i].get_state()
        return states

    def get_cluster_size(self):
        return self.cluster_size