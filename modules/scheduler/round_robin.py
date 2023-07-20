
from config import config
from modules.scheduler import Scheduler

class RoundRobinScheduler(Scheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = [(i - 1) % self.n for i in range(config.CLUSTERS_NUM)]

    def schedule(self) -> list:
        for i in range(len(self.cluster_assignment)):
            self.cluster_assignment[i] = (self.cluster_assignment[i] + 1) % self.n
        return self.cluster_assignment