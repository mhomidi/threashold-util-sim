from modules.applications import DistributedApplication
from modules.applications.queue import QueueApplication
from modules.utils.load_utils import LoadBalancer
from utils.distribution import Generator
import numpy as np


class DistQueueApp(DistributedApplication):
    def __init__(self, app_id, applications, arrival_generator: Generator, load_balancer: LoadBalancer, agents_len):
        super().__init__(app_id, applications)
        self.arrival_generator = arrival_generator
        self.loads = np.zeros(self.cluster_size)
        self.load_balancer = load_balancer
        self.agent_len = agents_len

    def update_dist_app(self, iteration, assignments):
        super().update_dist_app(iteration, assignments)

        self.assignments = assignments
        arrivals = int(self.arrival_generator.generate() * (float(self.cluster_size) /
                                                            self.agent_len))
        per_queue_arrivals = self.load_balancer.balance_load(
            arrivals, self.loads)

        self.utility = 0
        for i, app in enumerate(self.applications):
            app: QueueApplication
            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.update_state(iteration)

            self.loads[i] = app.get_load()
            self.utility -= app.get_current_queue_length()

    def stop(self, path):
        return super().stop(path)
