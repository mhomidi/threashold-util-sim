import numpy as np
from modules.policies import Policy


class GFairPolicy(Policy):
    def get_demands(self, state):
        demands = np.zeros(len(state))
        demands[state > 0] = 1
        return demands

    def update_policy(self, old_state, action, reward, new_state):
        return

    def printable_action(self, state):
        return 0
    
    def stop(self, path, id):
        return
