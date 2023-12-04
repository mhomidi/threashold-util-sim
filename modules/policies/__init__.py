import numpy as np


class Policy:

    # get_demand should return preferences for GPUs
    def get_demands(self, state):
        raise NotImplementedError

    def update_policy(self, old_state, action, reward, new_state):
        raise NotImplementedError


class ACPolicy(Policy):
    def get_demands(self, state):
        raise NotImplementedError

    def update_policy(self, old_state, action, reward, new_state):
        raise NotImplementedError


class GandivaFairPolicy(Policy):
    def get_demands(self, state):
        demands = np.zeros(len(state))
        for i, s in enumerate(state):
            demands[i] = s > 0
        return demands

    def update_policy(self, old_state, action, reward, new_state):
        return
