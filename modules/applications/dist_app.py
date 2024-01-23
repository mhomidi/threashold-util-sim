from modules.applications import DistributedApplication
from modules.applications.queue import QueueApplication
from modules.utils.load_utils import LoadBalancer
from utils.distribution import Generator
import numpy as np


class DistQueueApp(DistributedApplication):
    def __init__(self, app_id, applications, arrival_generator: Generator, load_balancer: LoadBalancer, queue_app_type='wo_dd'):
        super().__init__(app_id, applications)
        self.applications: list[QueueApplication]
        self.arrival_generator = arrival_generator
        self.arrival_history = list()
        self.loads = np.zeros(self.cluster_size)
        self.load_balancer = load_balancer
        self.assignments_history = list()
        self.queue_app_type = queue_app_type
        # self.app_dep_rates = np.array([app.departure_generator.rate for app in self.applications])

    def update_dist_app(self, iteration, assignments):
        super().update_dist_app(iteration, assignments)
        self.assignments_history.append(assignments)
        current_queue_lengths = list()
        for i, app in enumerate(self.applications):
            current_queue_lengths.append(app.get_current_queue_length())

        self.assignments = assignments
        arrivals = self.arrival_generator.generate()
        self.arrival_history.append(arrivals)
        app_dep_rates = np.array([app.departure_generator.rate for app in self.applications])
        per_queue_arrivals = self.load_balancer.balance_load(arrivals, current_queue_lengths, app_dep_rates)

        self.utility = 0
        for i, app in enumerate(self.applications):
            app: QueueApplication
            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.update_state(iteration)
            if self.queue_app_type == 'without_deadline':
                self.utility -= app.get_current_queue_length()
            elif self.queue_app_type == 'with_deadline':
                self.utility += app.get_imm_throughput()

    def stop(self, path):
        super().stop(path)
        app_path = path + '/agent_' + str(self.app_id)
        np.savetxt(app_path + '/more_data.csv',
                   np.array([self.utility_history, self.arrival_history]).T, 
                   fmt='%d', delimiter=',')
        np.savetxt(app_path + '/assigns.csv',
                   np.array(self.assignments_history), 
                   fmt='%d', delimiter=',')

    
