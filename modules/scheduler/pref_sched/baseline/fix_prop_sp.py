from modules.scheduler.pref_sched import PrefScheduler
import config as config
import numpy as np


class FixedProportionalSpaceSlicingScheduler(PrefScheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = np.array(
            [i % self.n for i in range(config.get('cluster_num'))])

    def schedule(self) -> list:
        self.dispatcher.set_cluster_assignments(
            self.cluster_assignment.tolist())
        return self.cluster_assignment.tolist()
