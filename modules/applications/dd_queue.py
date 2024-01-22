from modules.applications.queue import QueueApplication
from utils.distribution import Generator
from modules.utils.load_utils import *


class DeadlineQueueApplication(QueueApplication):

    def __init__(self, max_queue_length, departure_generator: Generator, alpha, deadline):
        super().__init__(max_queue_length, departure_generator, alpha)
        self.arrivals = np.zeros(deadline + 1)

    def set_arrival(self, arrival):
        self.arrivals = np.roll(self.arrivals, 1)
        self.arrivals[0] = arrival
        self.queue_length = self.arrivals.sum()
        self.arrival = arrival
        self.avg_arrival_rate = self.alpha * self.avg_arrival_rate + (1 - self.alpha) * self.arrival
        self.arrival_history.append(arrival)

    def update_state(self, iteration):
        self.state_history.append(self.state)
        self.queue_length_history.append(self.queue_length)

        departure = self.departure_generator.generate()
        self.departure_history.append(departure)
        self.departure = min(self.queue_length, departure * self.assignment)
        self.avg_throughput *= self.alpha
        self.avg_throughput += ((1 - self.alpha) * self.departure)
        self.corr_avg_throughput = self.avg_throughput / (1 - self.alpha ** (iteration + 1))
        self.update_arrivals_array()
        self.queue_length = self.arrivals.sum()
        self.state = min(self.max_queue_length, self.queue_length)

    def update_arrivals_array(self):
        dep = self.departure
        for idx in range(len(self.arrivals))[::-1]:
            if self.arrivals[idx] >= dep:
                self.arrivals[idx] -= dep
                dep = 0
            else:
                dep -= self.arrivals[idx]
                self.arrivals[idx] = 0
            if dep == 0:
                break
