from modules.scheduler.pref_sched import PrefScheduler
import config as config
import numpy as np


class LottoryTimeSlicingScheduler(PrefScheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = np.array(
            [-1 for _ in range(config.get('cluster_num'))])

    def schedule(self) -> list:
        rand = np.random.randint(0, self.n)
        self.cluster_assignment[:] = rand
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment.tolist()
