from modules.scheduler import Scheduler
import config as config

class FixedAgentScheduler(Scheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = [0 for _ in range(config.get('cluster_num'))]

    def schedule(self) -> list:
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment