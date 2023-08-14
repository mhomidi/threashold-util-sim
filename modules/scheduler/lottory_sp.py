from modules.scheduler import Scheduler
import config as config
import numpy as np

class LottorySpaceSlicingScheduler(Scheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = [-1 for _ in range(config.get('cluster_num'))]

    def schedule(self) -> list:
        self.cluster_assignment = np.random.randint(0, self.n, config.get('cluster_num')).tolist()
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment