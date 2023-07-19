

from config.config import *


class Scheduler:

    def __init__(self) -> None:
        self.core_assignment = [-1 for i in range(CLUSTERS_NUM)]
        self.report = None

    def schedule(self) -> list:
        raise NotImplementedError()
    
    def dist_tokens(self) -> None:
        raise NotImplementedError()

    def get_new_budgets(self):
        new_budgets = list()
        for agent in self.report:
            new_budgets.append(agent[0])
        return new_budgets
        
    def set_report(self, report: list) -> None:
        self.report = report