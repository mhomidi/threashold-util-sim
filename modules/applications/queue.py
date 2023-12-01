from modules.applications import Application, State
from utils.distribution import *
import numpy as np


class QueueApplication(Application):

    def __init__(self, throughput, max_queue_length, initial_queue_length) -> None:
        super().__init__()
        self.num_clusters = config.get(config.CLUSTER_NUM)
        self.init_state = State(name="init", val=initial_queue_length)
        self.curr_queue_lengths = np.array(self.init_state.val)
        self.max_queue_length = max_queue_length
        self.curr_state = np.maximum(self.max_queue_length, self.curr_queue_lengths)
        self.departure_generators = [PoissonGenerator(throughput[i]) for i in range(0, self.num_clusters)]
        self.arrivals = np.zeros(self.num_clusters)
        self.allocations = np.zeros(self.num_clusters)

    def set_arrival(self, arrivals):
        self.arrivals = arrivals

    def set_allocation(self, allocations):
        self.allocations = allocations

    def go_next_state(self):
        next_departures = np.array([a.generate() for a in self.departure_generators])
        self.curr_queue_lengths = np.minimum(self.curr_queue_lengths + self.arrivals -
                                             self.allocations * next_departures, 0)
        self.curr_state = np.maximum(self.max_queue_length, self.curr_queue_lengths)
    
    def get_current_queue_lengths(self):
        return self.curr_queue_lengths

    def get_utilities(self):
        return -self.curr_queue_lengths
