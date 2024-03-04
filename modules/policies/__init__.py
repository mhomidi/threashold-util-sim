class Policy:

    def __init__(self, num_nodes) -> None:
        self.num_nodes = num_nodes

    # get_demand should return preferences for GPUs
    def get_demands(self, state):
        return state

    def update_policy(self, old_state, action, reward, new_state):
        return

    def printable_action(self, state):
        return 0

    def stop(self, path, agent_id):
        return
