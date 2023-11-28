
from modules.scheduler.queue_sched import QueueBaseScheduler
import cvxpy as cp
import numpy as np
import config


class FinishTimeFairnessScheduler(QueueBaseScheduler):

    def schedule(self) -> list:
        a_num = config.get('default_agent_num')
        c_num = config.get('cluster_num')
        q_lengths = self.get_q_lengths()
        arrival_rate = 0.5
        q_lengths += arrival_rate
        g_ex = self.get_g_ex()
        throughputs = self.get_throughput()
        g_sh = cp.Variable((a_num, c_num), boolean=True)
        ones = np.ones(c_num)
        constraint = [g_sh.sum(axis=0) == ones]
        rhos = list()
        for agent_id in range(a_num):
            throughput_ex = (throughputs[agent_id] * g_ex[agent_id]).sum()
            throughput_sh = (cp.multiply(
                throughputs[agent_id], g_sh[agent_id])).sum()
            t_ex = q_lengths[agent_id] - \
                np.minimum(q_lengths[agent_id], throughput_ex) + 1e-3
            t_sh = q_lengths[agent_id] - \
                cp.minimum(q_lengths[agent_id], throughput_sh)
            rho = t_sh / t_ex
            rhos.append(rho)
        solver = cp.Problem(cp.Minimize(cp.maximum(*rhos)),
                            constraints=constraint)
        solver.solve(qcp=True)
        return self.get_allocation(np.array((np.array(g_sh.value) > 0.5), dtype=int)).tolist()

    def get_allocation(self, one_hot_alloc: np.ndarray) -> np.ndarray:
        return np.argmax(one_hot_alloc, axis=0)

    def get_q_lengths(self) -> np.ndarray:
        ls = [item['q_length'] for item in self.report]
        return np.array(ls).T

    def get_g_ex(self) -> np.ndarray:
        a_num = config.get('default_agent_num')
        c_num = config.get('cluster_num')
        gs = [[0. for _ in range(c_num)]
              for _ in range(a_num)]

        # TODO: assume that c_num >= agent_num. We can read from a json file.
        for i in range(c_num):
            a_id = i % a_num
            gs[a_id][i] = 1.

        return np.array(gs)

    def get_throughput(self) -> np.ndarray:
        throughputs = [item['throughput'] for item in self.report]
        return np.array(throughputs)
