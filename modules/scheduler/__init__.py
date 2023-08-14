

from modules.dispatcher import Dispatcher
import config
import numpy as np


class Scheduler:

    def __init__(self) -> None:
        self.report = None
        self.cluster_assignment = [-1 for _ in range(config.get('cluster_num'))]
        self.dispatcher = Dispatcher()
        self.assignment_history = list()

    def schedule(self) -> list:
        raise NotImplementedError()

    def dist_tokens(self) -> None:
        budgets = [config.get('budget') for _ in range(self.n)]
        self.dispatcher.set_budgets(budgets)
        self.dispatcher.set_dist_token([0. for _ in range(self.n)])

    def set_report(self, report: list) -> None:
        self.report = report

    def get_report(self) -> list:
        return self.report

    def get_dispatcher(self) -> Dispatcher:
        return self.dispatcher
    
    def get_cluster_assignments(self):
        return self.cluster_assignment
    
    def run(self):
        for episode in range(config.get('episodes')):
            self.dispatcher.recieve_data()

            self.set_report(self.dispatcher.get_report())
            self.assignment_history.append(self.schedule())
            self.dist_tokens()
            self.dispatcher.send_data()
            if episode % 500 == 499:
                print("episode {e} done".format(e=episode + 1))


class TokenBaseScheduler(Scheduler):

    def __init__(self) -> None:
        super().__init__()
        self.token_distribution = np.zeros(config.get('token_dist_sample'))
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
        self.dispatcher.set_dist_token(self.token_distribution.tolist())
        self.time += 1  

    def get_new_budgets(self):
        new_budgets = list()
        for agent in self.report:
            new_budgets.append(agent[0])
        return new_budgets
    
    def get_token_distribution(self):
        return self.token_distribution.tolist()
    
    def update_token_distribution(self, budgets: list):
        budget_count = [0 for _ in range(config.get('token_dist_sample'))]
        for b in budgets:
            if int(config.get('budget') - config.get('token_dist_sample')) / 2 < b < int(config.get('budget') + config.get('token_dist_sample') / 2):
                budget_count[b - int(config.get('budget') - config.get('token_dist_sample') / 2)] += 1
            elif config.get('budget') - config.get('token_dist_sample') / 2 >= b:
                budget_count[0] += 1
            elif config.get('budget') + config.get('token_dist_sample') / 2 <= b:
                budget_count[-1] + 1
        
        budget_count = np.array(budget_count)
        self.token_distribution *= config.get('decay_factor') * (1 - config.get('decay_factor') ** (self.time - 1))
        self.token_distribution += ((1 - config.get('decay_factor')) * budget_count)
        self.token_distribution /= (1 - config.get('decay_factor') ** self.time)
        self.token_distribution /= config.get('default_agent_num')



