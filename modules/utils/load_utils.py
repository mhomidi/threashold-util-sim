import numpy as np


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
        avg_departure_rate[avg_departure_rate <= 1e-5] = 1e-5
        return queue_length / avg_departure_rate


class LoadBalancer:
    def __init__(self, load_calculator):
        self.load_calculator = load_calculator

    def balance_load(self, arrivals, current_q_lengths, avg_departure_rate):
        raise NotImplementedError


class RandomLoadBalancer(LoadBalancer):
    def balance_load(self, arrivals, current_q_lengths, avg_departure_rate):
        num_queues = len(current_q_lengths)
        per_queue_arrivals = np.zeros(num_queues)
        for i in range(0, arrivals):
            index = np.random.choice(range(0, num_queues))
            per_queue_arrivals[index] += 1
        return per_queue_arrivals


class PowerOfTwoChoices(LoadBalancer):
    def balance_load(self, arrivals, current_q_lengths, avg_departure_rate):
        num_queues = len(current_q_lengths)
        per_queue_arrivals = np.zeros(num_queues)
        if num_queues == 1:
            per_queue_arrivals[0] = arrivals
            return per_queue_arrivals

        if avg_departure_rate.sum() == 0:
            avg_departure_rate += 1

        probs = avg_departure_rate / avg_departure_rate.sum(axis=-1)
        probs[probs <= 0.] = 0
        probs /= probs.sum(axis=-1)

        if (probs > 0).sum() < 2:
            per_queue_arrivals[probs > 0] = arrivals
            return per_queue_arrivals

        for i in range(0, arrivals):
            new_q_lengths = per_queue_arrivals + current_q_lengths
            new_loads = self.load_calculator.calculate_load(new_q_lengths, avg_departure_rate)
            index1, index2 = np.random.choice(range(0, num_queues), size=2, replace=False, p=probs)
            if new_loads[index1] < new_loads[index2]:
                per_queue_arrivals[index1] += 1
            else:
                per_queue_arrivals[index2] += 1
        return per_queue_arrivals
