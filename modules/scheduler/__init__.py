

from config.config import *
from modules.dispatcher import Dispatcher


class Scheduler:

    def __init__(self) -> None:
        self.report = None
        self.cluster_assignment = [-1 for _ in range(CLUSTERS_NUM)]
        self.dispatcher = Dispatcher()

    def schedule(self) -> list:
        raise NotImplementedError()
        
    def set_report(self, report: list) -> None:
        self.report = report

    def get_report(self) -> list:
        return self.report

    def get_dispatcher(self) -> Dispatcher:
        return self.dispatcher
    
    def get_cluster_assignments(self):
        return self.cluster_assignment


class TokenBaseScheduler(Scheduler):
    
    def dist_tokens(self) -> None:
        raise NotImplementedError()

    def get_new_budgets(self):
        new_budgets = list()
        for agent in self.report:
            new_budgets.append(agent[0])
        return new_budgets

