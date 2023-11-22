
from utils.queue import *
import time
import config


class Dispatcher:
    def __init__(self) -> None:
        self.incoming_queues = list()
        self.out_queues = list()
        self.report = list()
        self.report_update = list()
        self.last_id = 0
        self.weights = list()

    def connect(self,
                   incoming_queue: AgentToDispatcherQueue,
                   out_queue: DispatcherToAgentQueue,
                   weight: float = 1. / config.get('default_agent_num')
                   ):
        self.incoming_queues.append(incoming_queue)
        self.out_queues.append(out_queue)
        self.report.append(None)
        self.report_update.append(False)
        self.weights.append(weight)
        incoming_queue.set_id(self.last_id)
        out_queue.set_id(self.last_id)
        self.last_id += 1

    def set_budgets(self, new_budgets: list) -> None:
        self.budgets = new_budgets
    
    def set_cluster_assignments(self, assignments: list) -> None:
        self.cluster_assignments = assignments

    def set_dist_token(self, dist: list) -> None:
        self.token_dist = dist
    
    def is_report_new(self):
        return sum(self.report_update) == len(self.report_update)

    def set_bid(self, agent_id: int, budget: int, bid: list) -> None:
        self.report[agent_id] = [budget, bid]
        self.report_update[agent_id] = True

    def get_report(self) -> list:
        return self.report

    def get_weights(self) -> list:
        return self.weights
    
    def send_data(self) -> None:
        for id, queue in enumerate(self.out_queues):
            queue: DispatcherToAgentQueue
            queue.put(self.budgets[id], self.cluster_assignments, self.token_dist)

    def recieve_data(self) -> None:
        while not self.is_report_new():
            time.sleep(0.0001)
            for id, queue in enumerate(self.incoming_queues):
                queue: AgentToDispatcherQueue
                if queue.is_empty():
                    continue
                data = queue.get()
                id = data[0]
                budget = data[1]
                pref = data[2]
                self.set_bid(id, budget, pref)
                
        self.report_update = [False for _ in range(len(self.incoming_queues))]
        

            

    