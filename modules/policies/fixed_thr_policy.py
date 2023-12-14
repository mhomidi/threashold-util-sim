import numpy as np

from modules.policies import Policy


class FixedThrPolicy(Policy):
    def __init__(self, num_clusters, threshold):
        super().__init__(num_clusters)
        self.threshold = threshold

    def get_demands(self, state):
        demands = state.copy()
        demands[state < self.threshold] = 0
        return demands

    def update_policy(self, old_state, action, reward, new_state):
        return

    def printable_action(self, state):
        return self.threshold
