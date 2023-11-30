
from modules.scheduler.queue_sched import QueueBaseScheduler
import numpy as np
import config


class GandivaScheduler(QueueBaseScheduler):

    def __init__(self) -> None:
        super().__init__()
        self.weights: np.ndarray = None
        self.is_first_schedule = True

    def schedule(self) -> list:
        if self.is_first_schedule:
            self.read_weights()
            self.trade_resources()

    def read_weights(self):
        w_l = list()
        for data in self.report:
            w_l.append(data['weight'])
        self.weights = np.array([[w_l[j] for i in range(config.get(
            config.CLUSTER_NUM))] for j in range(config.get(config.AGENT_NUM))])

    def trade_resources(self):
        throughputs = np.array([data['throughput'] for data in self.report])
        for low in range(config.get(config.CLUSTER_NUM)):
            high = config.get(config.CLUSTER_NUM) - 1
            while high > low:
                while True:
                    f_max_index, s_max_index, min_index = self.get_available_indices(
                        throughputs.copy(), 0, high)
                    if f_max_index == min_index or f_max_index == None or min_index == None:
                        break
                    speed_up = throughputs[s_max_index][high] / \
                        throughputs[s_max_index][low]
                    self.trade(f_max_index, min_index,
                               speed_up, low=low, high=high)
                high -= 1

    def trade(self, f_max_index: int, min_index: int, speed_up: float, low: int = 0, high: int = config.get(config.CLUSTER_NUM) - 1):
        low_f_max_weight = self.weights[f_max_index][low]
        high_min_weight = self.weights[min_index][high]

        if low_f_max_weight / speed_up > high_min_weight:
            self.weights[f_max_index][low] -= high_min_weight * speed_up
            self.weights[min_index][low] += high_min_weight * speed_up
            self.weights[f_max_index][high] += high_min_weight
            self.weights[min_index][high] = 0.0
        else:
            self.weights[f_max_index][low] = 0.0
            self.weights[min_index][low] += low_f_max_weight
            self.weights[f_max_index][high] += low_f_max_weight / speed_up
            self.weights[min_index][high] -= low_f_max_weight / speed_up

    def get_available_indices(self, throughput: np.ndarray, low_util_index: int, high_util_index: int) -> tuple:
        f_max_index = None
        s_max_index = None
        min_index = None
        high_util_throughput = throughput[:, high_util_index]
        low_util_throughput = throughput[:, low_util_index]
        high_util_throughput = high_util_throughput / low_util_throughput
        low_util_throughput = np.ones(low_util_throughput.shape)
        sorted_args = np.argsort(high_util_throughput).tolist()
        for arg in sorted_args:
            if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
                min_index = arg
                break
        for idx, arg in enumerate(sorted_args[::-1]):
            if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
                f_max_index = arg
                if idx < len(sorted_args) - 1:
                    s_max_index = sorted_args[::-1][idx + 1]
                break
        return f_max_index, s_max_index, min_index
