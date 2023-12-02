from modules.applications import Application
from utils.distribution import *


class QueueApplication(Application):

    # We use current_state only for training NN - otherwise, current_queue_length should be used
    def __init__(self, max_queue_length, departure_generator, avg_throughput_alpha) -> None:
        super().__init__()
        self.init_state = 0
        self.curr_queue_length = 0
        self.max_queue_length = max_queue_length
        self.curr_state = 0
        self.departure_generator = departure_generator
        self.arrival = 0
        self.assignment = 0
        self.avg_throughput = 0
        self.avg_throughput_alpha = avg_throughput_alpha
        self.departure = 0

    def set_arrival(self, arrival):
        self.arrival = arrival

    def set_assignment(self, assignment):
        self.assignment = assignment

    def update_state(self):
        self.state_history.append(self.get_current_queue_length())
        self.departure = min(self.curr_queue_length + self.arrival,
                             self.departure_generator.generate() * self.assignment)
        self.avg_throughput *= (1 - self.avg_throughput_alpha)
        self.avg_throughput += (self.avg_throughput_alpha * self.departure)
        self.curr_queue_length = self.curr_queue_length + self.arrival - self.departure
        self.curr_state = max(self.max_queue_length, self.curr_queue_length)

    def get_current_queue_length(self):
        return self.curr_queue_length

    def get_imm_throughput(self):
        return self.departure

    def get_avg_throughput(self):
        return self.avg_throughput
