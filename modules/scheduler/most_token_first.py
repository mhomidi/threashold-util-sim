

from modules.scheduler import TokenBaseScheduler

from config import config


class MostTokenFirstScheduler(TokenBaseScheduler):
    def __init__(self) -> None:
        super(MostTokenFirstScheduler, self).__init__()
        self.last_random_assign_rr_turn = 0

    def schedule(self) -> list:
        self.cluster_assignment = [-1 for i in range(config.CLUSTERS_NUM)]
        self.gathered_token = 0
        for _ in range(config.CLUSTERS_NUM):
            max_budget_agent_index = self.find_max_budget_agent_index()
            if max_budget_agent_index == -1:
                self.assgin_randomly_available_clusters()
                break
            agent_pref = self.report[max_budget_agent_index][1][0]
            self.cluster_assignment[agent_pref] = max_budget_agent_index
            self.report[max_budget_agent_index][0] -= 1
            self.gathered_token += 1

            for agent in self.report:
                try:
                    agent[1].remove(agent_pref)
                except:
                    pass
        self.dispatcher.set_cluster_assignments(self.cluster_assignment)
        return self.cluster_assignment    

    def find_max_budget_agent_index(self) -> int:
        max_b = -1
        max_b_idx = -1
        for index, agent in enumerate(self.report):
            if agent[0] > 0 and agent[0] > max_b and len(agent[1]) > 0:
                max_b = agent[0]
                max_b_idx = index
        return max_b_idx
    
    def assgin_randomly_available_clusters(self):
        n = len(self.report)
        for i, agent in enumerate(self.cluster_assignment):
            if agent == -1:
                self.last_random_assign_rr_turn %= n
                self.cluster_assignment[i] = self.last_random_assign_rr_turn
                self.last_random_assign_rr_turn += 1