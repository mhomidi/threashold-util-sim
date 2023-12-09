import numpy as np


class LoadBalancer:
    def balance_load(self, arrivals, current_loads):
        raise NotImplementedError


class RandomLoadBalancer(LoadBalancer):
    def balance_load(self, arrivals, current_loads):
        num_queues = len(current_loads)
        per_queue_arrivals = np.zeros(num_queues)
        for i in range(0, arrivals):
            index = np.random.choice(range(0, num_queues))
            per_queue_arrivals[index] += 1
        return per_queue_arrivals


class LoadCalculator:
    def calculate_load(self, queue_length, avg_departure_rate):
        raise NotImplementedError


class GFairLoadCalculator(LoadCalculator):
    def calculate_load(self, queue_length, avg_departure_rate):
        return queue_length


class ExpectedWaitTimeLoadCalculator(LoadCalculator):
    def calculate_load(self, queue_length, avg_departure_rate):
        return queue_length / avg_departure_rate
