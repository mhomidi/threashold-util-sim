

from config.config import *


class Scheduler:

    def __init__(self) -> None:
        self.report = None
        self.cluster_assignment = [-1 for i in range(CLUSTERS_NUM)]

    def schedule(self) -> list:
        raise NotImplementedError()
        
    def set_report(self, report: list) -> None:
        self.report = report


class TokenBaseScheduler(Scheduler):
    
    def dist_tokens(self) -> None:
        raise NotImplementedError()

    def get_new_budgets(self):
        new_budgets = list()
        for agent in self.report:
            new_budgets.append(agent[0])
        return new_budgets

