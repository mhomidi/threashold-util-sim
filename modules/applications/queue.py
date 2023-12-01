from modules.applications import Application, State
from utils.distribution import *


class QueueApplication(Application):

    def __init__(self, throughput, max_queue_length, initial_queue_length) -> None:
        super().__init__()
        self.init_state = State(name="init", val=initial_queue_length)
        self.curr_queue_length = self.init_state.val
        self.max_queue_length = max_queue_length
        self.curr_state = max(self.max_queue_length, self.curr_queue_length)
        self.departure_generator = PoissonGenerator(throughput)
        self.arrival = 0
        self.allocation = 0

    def set_arrival(self, arrival):
        self.arrival = arrival

    def set_allocation(self, allocation):
        self.allocation = allocation

    def go_next_state(self):
        next_departure = self.departure_generator.generate()
        self.curr_queue_length = max(self.curr_queue_length + self.arrival -
                                     self.allocation * next_departure, 0)
        self.curr_state = max(
            self.max_queue_length, self.curr_queue_length)

    def get_current_queue_length(self):
        return self.curr_queue_length

    def get_utility(self):
        return -self.curr_queue_length
