import config
from modules.applications.queue import QueueApplication
import numpy as np


class DistributedApplication:
    def __init__(self, applications):
        self.applications = applications
        self.cluster_size = config.get(config.CLUSTER_NUM)
        assert(self.cluster_size == len(applications))
        self.assignments = np.zeros(self.cluster_size)
        self.utility = 0
        self.assignments_history = list()
        self.utility_history = list()

    def update_dist_app(self, iteration, assignments):
        self.assignments_history.append(self.assignments)
        self.utility_history.append(self.utility)

    def get_utility(self):
        return self.utility

    def get_curr_state(self):
        states = np.zeros(self.cluster_size)
        for i in range(0, self.cluster_size):
            states[i] = self.applications[i].get_curr_state()
        return states


class DistQueueApp(DistributedApplication):
    def __init__(self, applications, arrival_generator, load_calculator, load_balancer):
        super().__init__(applications)
        self.applications: QueueApplication
        self.arrival_generator = arrival_generator
        self.loads = np.zeros(self.cluster_size)
        self.loads_history = list()
        self.load_calculator = load_calculator
        self.load_balancer = load_balancer
        self.queue_lengths = np.zeros(self.cluster_size)
        self.avg_departure_rates = np.zeros(self.cluster_size)

    def update_dist_app(self, iteration, assignments):
        super().update_dist_app(iteration, assignments)
        self.loads_history.append(self.loads)

        self.assignments = assignments
        arrivals = self.arrival_generator.generate()
        per_queue_arrivals = self.load_balancer.balance_load(arrivals, self.loads)

        self.utility = 0
        for i, app in enumerate(self.applications):

            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.update_state()

            self.queue_lengths[i] = app.get_current_queue_length()
            self.avg_departure_rates[i] = app.get_avg_throughput()
            self.utility += app.get_imm_throughput()

        self.loads = self.load_calculator.calculate_load(self.queue_lengths, self.avg_departure_rates)