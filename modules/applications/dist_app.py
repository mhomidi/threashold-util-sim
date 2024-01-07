from modules.applications import DistributedApplication
from modules.applications.queue import QueueApplication
from modules.utils.load_utils import LoadBalancer
from utils.distribution import Generator
import numpy as np


class DistQueueApp(DistributedApplication):
    def __init__(self, app_id, applications, arrival_generator: Generator, load_balancer: LoadBalancer):
        super().__init__(app_id, applications)
        self.applications: list[QueueApplication]
        self.arrival_generator = arrival_generator
        self.loads = np.zeros(self.cluster_size)
        self.load_balancer = load_balancer
        self.app_dep_rates = np.array([app.departure_generator.rate for app in self.applications])

    def update_dist_app(self, iteration, assignments):
        super().update_dist_app(iteration, assignments)
        current_queue_lengths = list()
        for i, app in enumerate(self.applications):
            current_queue_lengths.append(app.get_current_queue_length())

        self.assignments = assignments
        arrivals = self.arrival_generator.generate()
        per_queue_arrivals = self.load_balancer.balance_load(arrivals, current_queue_lengths, self.app_dep_rates)

        self.utility = 0
        for i, app in enumerate(self.applications):
            app: QueueApplication
            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.update_state(iteration)
            self.utility -= app.get_current_queue_length()
