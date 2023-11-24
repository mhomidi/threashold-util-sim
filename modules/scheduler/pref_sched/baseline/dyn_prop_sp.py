from modules.scheduler.pref_sched import PrefScheduler
import config as config


# Should used with fixed policy 0.0
class DynamicProportionalSpaceSlicingScheduler(PrefScheduler):

    def __init__(self, agent_n: int) -> None:
        super().__init__()
        self.n = agent_n
        self.cluster_assignment = [
            i % self.n for i in range(config.get('cluster_num'))]
        self.turn = 0

    def schedule(self) -> list:
        self.cluster_assignment = [
            -1 for i in range(config.get('cluster_num'))]
        self.gathered_token = 0
        for i in range(config.get('cluster_num')):
            max_budget_agent_index = self.turn % self.n
            self.turn += 1
            agent_pref = self.report[max_budget_agent_index][1][0]
            self.cluster_assignment[agent_pref] = max_budget_agent_index
            for agent in self.report:
                try:
                    agent[1].remove(agent_pref)
                except:
                    pass
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment
