from modules.applications import Application
from utils.distribution import *


class QueueApplication(Application):

    # We use current_state only for training NN - otherwise, current_queue_length should be used
    def __init__(self, max_queue_length, departure_generator) -> None:
        super().__init__()
        self.init_state = 0
        self.curr_queue_length = 0
        self.max_queue_length = max_queue_length
        self.curr_state = 0
        self.departure_generator = departure_generator
        self.arrival = 0
        self.assignment = 0
        self.itr = 0
        self.observed_departures = 0

    def set_arrival(self, arrival):
        self.arrival = arrival

    def set_assignment(self, allocation):
        self.assignment = allocation

    def go_next_state(self):
        self.states.append(self.get_current_queue_length())
        next_departure = self.departure_generator.generate() * self.assignment
        self.observed_departures += next_departure
        self.curr_queue_length = max(self.curr_queue_length + self.arrival - next_departure, 0)
        self.curr_state = max(self.max_queue_length, self.curr_queue_length)
        self.itr += 1

    def get_current_queue_length(self):
        return self.curr_queue_length

    def get_utility(self):
        # return -self.curr_queue_length
        return -self.curr_state

    def get_departure_rate(self):
        return self.observed_departures / self.itr
