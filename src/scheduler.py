
DIST_SAMPLE_NUMBER = 1000
DEFAULT_N = 10
DEFAULT_TOTAL_BUDGET = 100
CLUSTERS_NUMBER = 10
EPSILON = 1e-3


class Scheduler:
    def __init__(self) -> None:
        self.core_assignment = [-1 for i in range(CLUSTERS_NUMBER)]
        self.report = None

    def schedule(self) -> list:
        for cluster_id in range(CLUSTERS_NUMBER):
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

    def find_max_budget_agent_index(self) -> int:
        max_b = -1
        max_b_idx = -1
        for index, agent in enumerate(self.report):
            if agent[0] > max_b and len(agent[1]) > 0:
                max_b = agent[0]
                max_b_idx = index
        return max_b_idx
    
    def set_report(self, report: list) -> None:
        self.report = report