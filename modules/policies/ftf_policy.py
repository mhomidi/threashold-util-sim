import numpy as np
from modules.policies import Policy


class ThemisPolicy(Policy):
    def get_demands(self, state):
        return state

    def update_policy(self, old_state, action, reward, new_state):
        return

    def printable_action(self, state):
        return 0

    def stop(self, path, id):
        return
