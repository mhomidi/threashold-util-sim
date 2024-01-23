from modules.applications.queue import QueueApplication
from utils.distribution import Generator
from modules.utils.load_utils import *


class DeadlineQueueApplication(QueueApplication):

    def __init__(self, max_queue_length, departure_generator: Generator, alpha, deadline):
        super().__init__(max_queue_length, departure_generator, alpha)
        self.arrivals = np.zeros(deadline)

    def set_arrival(self, arrival):
        self.arrivals = np.roll(self.arrivals, 1)
        # self.arrivals[0] = arrival
        # self.queue_length = self.arrivals.sum()
        valid_arrivals = np.sum(self.arrivals) - self.arrivals[0]
        self.queue_length = min(self.queue_length, valid_arrivals)
        self.arrivals[0] = arrival
        self.arrival = arrival
        self.avg_arrival_rate = self.alpha * self.avg_arrival_rate + (1 - self.alpha) * self.arrival
        self.arrival_history.append(arrival)
