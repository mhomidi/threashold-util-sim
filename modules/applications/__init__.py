import numpy as np
import os


class Application:

    def __init__(self) -> None:
        self.state_history = list()
        self.state = None

    def update_state(self, iteration) -> None:
        raise NotImplementedError

    def get_state(self):
        return self.state

    def get_normalized_state(self):
        return self.state

    def stop(self, path, app_id):
        raise NotImplementedError


class DistributedApplication:
    def __init__(self, app_id, applications):
        self.app_id = app_id
        self.applications: list[Application] = applications
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

    def get_normalized_state(self):
        states = np.zeros(self.cluster_size)
        for i in range(0, self.cluster_size):
            states[i] = self.applications[i].get_normalized_state()
        return states

    def get_cluster_size(self):
        return self.cluster_size

    def stop(self, path):
        app_path = path + '/agent_' + str(self.app_id)
        if not os.path.exists(app_path):
            os.makedirs(app_path)
        for i, app in enumerate(self.applications):
            app.stop(app_path, i)
        np.savetxt(app_path + '/utility.csv',
                   self.utility_history, fmt='%d', delimiter=',')

    def get_more_data(self):
        return None
