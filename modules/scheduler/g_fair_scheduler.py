from modules.scheduler import Scheduler
import numpy as np


class GFairScheduler(Scheduler):

    class StrideScheduler:
        def __init__(self, weights):
            self.num_agents = len(weights)
            self.weights = weights
            self.passes = np.zeros(self.num_agents)
            self.strides = 100000 / self.weights

        def schedule(self, demands):
            agents_w_demands = np.arange(0, self.num_agents)[demands > 0]
            agent_id = -1
            if len(agents_w_demands) > 0:
                active_passes = self.passes[agents_w_demands]
                agent_id = agents_w_demands[np.argmin(active_passes)]
                self.passes[agent_id] += self.strides[agent_id]
            return agent_id

    def __init__(self, agent_weights, num_agents, num_clusters):
        super().__init__(agent_weights, num_agents, num_clusters)
        # self.speed_ups = np.zeros((self.num_agents, self.num_clusters))
        self.stride_schedulers = [self.StrideScheduler(self.agent_weights) for _ in range(self.num_clusters)]

    def run_scheduler(self, iteration, demands):
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        for i in range(self.num_clusters):
            agent_id = self.stride_schedulers[i].schedule(demands[:, i])
            if agent_id >= 0:
                self.assignments[(agent_id, i)] = 1

        return self.assignments, None

    # def trade_resources(self):
    #     for low in range(config.get(config.CLUSTER_NUM)):
    #         high = config.get(config.CLUSTER_NUM) - 1
    #         while high > low:
    #             while True:
    #                 f_max_index, s_max_index, min_index = self.get_available_indices(
    #                     self.speed_ups.copy(), low, high)
    #                 if f_max_index == min_index or f_max_index == None or min_index == None:
    #                     break
    #                 speed_up = self.speed_ups[s_max_index][high] / \
    #                     self.speed_ups[s_max_index][low]
    #                 self.trade(f_max_index, min_index,
    #                            speed_up, low=low, high=high)
    #             high -= 1
    #
    # def trade(self, f_max_index: int, min_index: int, speed_up: float, low: int = 0, high: int = config.get(config.CLUSTER_NUM) - 1):
    #     low_f_max_weight = self.weights[f_max_index][low]
    #     high_min_weight = self.weights[min_index][high]
    #
    #     if low_f_max_weight / speed_up > high_min_weight:
    #         self.weights[f_max_index][low] -= high_min_weight * speed_up
    #         self.weights[min_index][low] += high_min_weight * speed_up
    #         self.weights[f_max_index][high] += high_min_weight
    #         self.weights[min_index][high] = 0.0
    #     else:
    #         self.weights[f_max_index][low] = 0.0
    #         self.weights[min_index][low] += low_f_max_weight
    #         self.weights[f_max_index][high] += low_f_max_weight / speed_up
    #         self.weights[min_index][high] -= low_f_max_weight / speed_up
    #
    # def get_available_indices(self, throughput: np.ndarray, low_util_index: int, high_util_index: int) -> tuple:
    #     f_max_index = None
    #     s_max_index = None
    #     min_index = None
    #     high_util_throughput = throughput[:, high_util_index]
    #     low_util_throughput = throughput[:, low_util_index]
    #     high_util_throughput = high_util_throughput / low_util_throughput
    #     low_util_throughput = np.ones(low_util_throughput.shape)
    #     sorted_args = np.argsort(high_util_throughput).tolist()
    #     for arg in sorted_args:
    #         if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
    #             min_index = arg
    #             break
    #     for idx, arg in enumerate(sorted_args[::-1]):
    #         if self.weights[arg, low_util_index] > 0.0 and self.weights[arg, high_util_index] > 0.0:
    #             f_max_index = arg
    #             if idx < len(sorted_args) - 1:
    #                 s_max_index = sorted_args[::-1][idx + 1]
    #             break
    #     return f_max_index, s_max_index, min_index
