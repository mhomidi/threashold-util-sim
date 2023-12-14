class Policy:

    def __init__(self, num_clusters) -> None:
        self.num_clusters = num_clusters

    # get_demand should return preferences for GPUs
    def get_demands(self, state):
        raise NotImplementedError

    def update_policy(self, old_state, action, reward, new_state):
        raise NotImplementedError

    def printable_action(self, state):
        raise NotImplementedError
