
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
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment
    

    def dist_tokens(self) -> None:
        budgets = [config.BUDGET for _ in range(self.n)]
        self.dispatcher.set_budgets(budgets)
        self.dispatcher.set_dist_token([0. for _ in range(self.n)])