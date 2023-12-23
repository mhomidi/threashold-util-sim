from modules.applications import Application
from utils.distribution import Generator
from modules.utils.load_utils import *

DEPART_ALPHA = 0.98


class QueueApplication(Application):

    # We use current_state only for training NN - otherwise, current_queue_length should be used
    def __init__(self, max_queue_length, departure_generator: Generator,
                 avg_throughput_alpha, load_calculator: LoadCalculator):
        super().__init__()
        self.init_state = 0
        self.queue_length = 0
        self.queue_length_history = list()
        self.max_queue_length = max_queue_length
        self.state = 0
        self.departure_generator = departure_generator
        self.arrival = 0
        self.assignment = 0
        self.assignment_history = list()
        self.avg_throughput = 0
        self.avg_throughput_alpha = avg_throughput_alpha
        self.actual_departure = 0
        self.expected_departure = 0
        self.load = 0
        self.loads_history = list()
        self.state_history = list()
        self.load_calculator = load_calculator

    def set_arrival(self, arrival):
        self.arrival = arrival

    def set_assignment(self, assignment):
        self.assignment_history.append(self.assignment)
        self.assignment = assignment

    def update_state(self, iteration):
        self.state_history.append(self.state)
        self.queue_length_history.append(self.queue_length)
        self.loads_history.append(self.load)
        departure = self.departure_generator.generate()
        self.actual_departure = min(self.queue_length + self.arrival,
                                    departure * self.assignment)
        # self.expected_departure = (self.expected_departure * DEPART_ALPHA) + ((1 - DEPART_ALPHA) * min(departure, self.queue_length + self.arrival))
        self.expected_departure = min(
            departure, self.queue_length + self.arrival)
        self.avg_throughput *= (1 - self.avg_throughput_alpha)
        self.avg_throughput += (self.avg_throughput_alpha *
                                self.actual_departure)
        avg_throughput = self.avg_throughput + 1e-3
        avg_throughput /= (1 - (1 -
                                self.avg_throughput_alpha) ** (iteration + 1))
        self.queue_length = self.queue_length + self.arrival - self.actual_departure
        self.state = min(self.max_queue_length, self.queue_length)
        self.queue_length = self.state
        self.load = self.load_calculator.calculate_load(
            self.queue_length, avg_throughput)

    def get_current_queue_length(self):
        return self.queue_length

    def get_imm_throughput(self):
        return self.actual_departure

    def get_avg_throughput(self):
        return self.avg_throughput

    def get_exp_departure(self):
        return self.expected_departure

    def get_load(self):
        return self.load

    def stop(self, path, id):
        data = np.array([self.queue_length_history, self.assignment_history,
                        self.state_history, self.loads_history]).T
        np.savetxt(path + "/app_" + str(id) + '.csv',
                   data, delimiter=',', fmt='%.2f')
