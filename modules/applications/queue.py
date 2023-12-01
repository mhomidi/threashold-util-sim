
from modules.applications import Application, State
from utils.distribution import *
import random
import numpy as np


class QueueApplication(Application):

    def __init__(self, throughput, max_queue_length, initial_queue_length) -> None:
        super().__init__()
        self.cluster_range = range(config.get(config.CLUSTER_NUM))
        self.init_state = State(name="init", val=initial_queue_length)
        self.curr_queue_lengths = np.array(self.init_state.val)
        self.max_queue_length = max_queue_length
        self.curr_state = np.maximum(self.max_queue_length, self.curr_queue_lengths)
        self.departure_generators = [PoissonGenerator(throughput[i]) for i in self.cluster_range]

    def go_next_state(self, arrivals, allocations) -> None:
        # self.curr_state.set_val([self.arrival_generator.generate()])
        # self.max_queue_length += self.init_state.val[0]

        # TODO: do it by numpy all
        next_departures = np.array([a.generate() for a in self.departure_generators])
        self.curr_queue_lengths = np.minimum(self.curr_queue_lengths + arrivals - allocations * next_departures, 0)
        self.curr_state = np.maximum(self.max_queue_length, self.curr_queue_lengths)
    
    def get_current_queue_lengths(self) -> list:
        return self.curr_queue_lengths

    # TODO: remove this
    def reduce_length(self, passed_job: int):
        self.max_queue_length -= passed_job
        if self.max_queue_length < 0:
            self.max_queue_length = 0

    def get_throughput(self) -> list:
        # TODO: -self.curr_queue_length
        return self.throughput
    
    def get_utilities(self):
        return -self.curr_queue_lengths
