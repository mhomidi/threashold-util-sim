

from modules.scheduler import Scheduler

from config.config import *


class MostTokenFirstScheduler(Scheduler):
    def __init__(self) -> None:
        self.last_turn = 0
        super(MostTokenFirstScheduler, self).__init__()

    def schedule(self) -> list:
        self.core_assignment = [-1 for i in range(CLUSTERS_NUM)]
        for cluster_id in range(CLUSTERS_NUM):
            max_budget_agent_index = self.find_max_budget_agent_index()
            if max_budget_agent_index == -1:
                return self.core_assignment
            agent_pref = self.report[max_budget_agent_index][1][0]
            self.core_assignment[agent_pref] = max_budget_agent_index
            self.report[max_budget_agent_index][0] -= 1
            for agent in self.report:
                try:
                    agent[1].remove(agent_pref)
                except:
                    pass
        return self.core_assignment

    def dist_tokens(self) -> None:
        tokens = 0
        for a in self.core_assignment:
            if a != -1:
                tokens += 1
        n = len(self.report)
        while tokens > 0:
            self.report[self.last_turn % n][0] += 1
            tokens -= 1
            self.last_turn += 1

    def find_max_budget_agent_index(self) -> int:
        max_b = -1
        max_b_idx = -1
        for index, agent in enumerate(self.report):
            if agent[0] > 0 and agent[0] > max_b and len(agent[1]) > 0:
                max_b = agent[0]
                max_b_idx = index
        return max_b_idx