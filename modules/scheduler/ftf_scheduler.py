from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np


class FinishTimeFairnessScheduler(Scheduler):

    def __init__(self, agent_weights, num_agents, num_clusters):
        super().__init__(agent_weights, num_agents, num_clusters)
        self.throughputs = np.zeros((num_agents, num_clusters))

    def set_more_data(self, data):
        self.throughputs = data

    def run_scheduler(self, iteration, demands) -> list:
        q_lengths = demands
        g_ex = self.get_g_ex()
        throughputs = self.throughputs.copy()
        g_sh = cp.Variable((self.num_agents, self.num_clusters), boolean=True)
        ones = np.ones(self.num_clusters)
        constraint = [g_sh.sum(axis=0) == ones]
        rhos = list()
        for agent_id in range(self.num_agents):
            for cluster_id in range(self.num_clusters):
                t_ex = q_lengths[agent_id][cluster_id] - \
                    throughputs[agent_id][cluster_id] * \
                    g_ex[agent_id][cluster_id] + 1e-3

                t_sh = q_lengths[agent_id][cluster_id] - cp.multiply(
                    throughputs[agent_id][cluster_id], g_sh[agent_id][cluster_id])
            # t_ex = q_lengths[agent_id] - \
            #     np.minimum(q_lengths[agent_id], throughput_ex) + 1e-3
            # t_sh = q_lengths[agent_id] - \
            #     cp.minimum(q_lengths[agent_id], throughput_sh)
                rho = t_sh / t_ex
                rhos.append(rho)
        solver = cp.Problem(cp.Minimize(cp.maximum(*rhos)),
                            constraints=constraint)
        solver.solve(qcp=True)
        return np.array((np.array(g_sh.value) > 0.5), dtype=int), None

    def get_allocation(self, one_hot_alloc: np.ndarray) -> np.ndarray:
        return np.argmax(one_hot_alloc, axis=0)

    def get_g_ex(self) -> np.ndarray:
        return np.ones((self.num_agents, self.num_clusters)) * self.agent_weights.reshape((self.num_agents, 1))
