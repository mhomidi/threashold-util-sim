import numpy as np


class LoadBalancer:
    @staticmethod
    def balance_load(arrivals, current_loads):
        raise NotImplementedError


class RandomLoadBalancer(LoadBalancer):
    @staticmethod
    def balance_load(arrivals, current_loads):
        num_queues = len(current_loads)
        per_queue_arrivals = np.zeros(num_queues)
        for i in range(0, arrivals):
            index = np.random.choice(range(0, num_queues))
            per_queue_arrivals[index] += 1
        return per_queue_arrivals


class LoadCalculator:
    @staticmethod
    def calculate_load(queue_length, avg_departure_rate):
        raise NotImplementedError


class GFairLoadCalculator(LoadCalculator):
    @staticmethod
    def calculate_load(queue_length, avg_departure_rate):
        return queue_length


class ExpectedWaitTimeLoadCalculator(LoadCalculator):
    @staticmethod
    def calculate_load(queue_length, avg_departure_rate):
        if avg_departure_rate == 0:
            return queue_length
        else:
            return queue_length / avg_departure_rate
