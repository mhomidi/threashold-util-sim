

from config import config
from modules.dispatcher import Dispatcher


class Scheduler:

    def __init__(self) -> None:
        self.report = None
        self.cluster_assignment = [-1 for _ in range(config.CLUSTERS_NUM)]
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

    def __init__(self) -> None:
        super().__init__()
        self.token_distribution = [0. for _ in range(config.TOKEN_DIST_SAMPLE)]
        self.last_token_rr_turn = 0
        self.gathered_token = 0
        self.time = 1
    
    def dist_tokens(self) -> None:
        tokens = self.gathered_token
        n = len(self.report)
        while tokens > 0:
            self.report[self.last_token_rr_turn % n][0] += 1
            tokens -= 1
            self.last_token_rr_turn += 1
        budgets = list()
        for row in self.report:
            budgets.append(row[0])
        self.dispatcher.set_budgets(budgets)
        self.update_token_distribution(budgets)
        self.dispatcher.set_dist_token(self.token_distribution)
        self.time += 1  

    def get_new_budgets(self):
        new_budgets = list()
        for agent in self.report:
            new_budgets.append(agent[0])
        return new_budgets
    
    def get_token_distribution(self):
        return self.token_distribution
    
    def update_token_distribution(self, budgets: list):
        budget_count = [0 for _ in range(config.TOKEN_DIST_SAMPLE)]
        for b in budgets:
            if int(config.BUDGET - config.TOKEN_DIST_SAMPLE) / 2 < b < int(config.BUDGET + config.TOKEN_DIST_SAMPLE / 2):
                budget_count[b - int(config.BUDGET - config.TOKEN_DIST_SAMPLE / 2)] += 1
            elif config.BUDGET - config.TOKEN_DIST_SAMPLE / 2 >= b:
                budget_count[0] += 1
            elif config.BUDGET + config.TOKEN_DIST_SAMPLE / 2 <= b:
                budget_count[-1] + 1

        for index, count in enumerate(budget_count):
            self.token_distribution[index] *= config.ALPHA * (1 - config.ALPHA ** (self.time - 1))
            self.token_distribution[index] += ((1 - config.ALPHA) * count)
            self.token_distribution[index] /= (1 - config.ALPHA ** self.time)
            self.token_distribution[index] /= config.DEFAULT_NUM_AGENT



