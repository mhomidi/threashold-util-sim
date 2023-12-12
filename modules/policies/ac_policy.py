from modules.policies import Policy
import numpy as np


class ACPolicy(Policy):

    def get_demands(self, state):
        demands = np.arange(self.num_clusters)
        np.random.shuffle(demands)
        return demands

    def update_policy(self, old_state, action, reward, new_state):
        # TODO: should be implemeted
        pass
