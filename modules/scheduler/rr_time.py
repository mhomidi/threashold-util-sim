from modules.scheduler import Scheduler
import config as config
import numpy as np

class RoundRobinTimeSlicingScheduler(Scheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = np.array([-1 for _ in range(config.get('cluster_num'))])

    def schedule(self) -> list:
        self.cluster_assignment = (self.cluster_assignment + 1) % self.n
        self.dispatcher.set_cluster_assignments(self.cluster_assignment.tolist())
        return self.cluster_assignment.tolist()