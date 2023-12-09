from modules.policies import Policy


class ACPolicy(Policy):
    def get_demands(self, state):
        raise NotImplementedError

    def update_policy(self, old_state, action, reward, new_state):
        raise NotImplementedError
    