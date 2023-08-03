
from utils.queue import *


class Dispatcher:
    def __init__(self) -> None:
        self.incoming_queues = list()
        self.out_queues = list()
        self.report = list()
        self.report_update = list()
        self.last_id = 0

    def connect(self,
                   incoming_queue: AgentToDispatcherQueue,
                   out_queue: DispatcherToAgentQueue
                   ):
        self.incoming_queues.append(incoming_queue)
        self.out_queues.append(out_queue)
        self.report.append(None)
        self.report_update.append(False)
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
        for i in range(self.last_id):
            if not self.report_update[i]:
                return False
        return True

    def set_bid(self, agent_id: int, budget: int, bid: list) -> None:
        self.report[agent_id] = [budget, bid]
        self.report_update[agent_id] = True

    def get_report(self) -> list:
        if not self.is_report_new():
            return None
        return self.report
    
    def send_data(self) -> None:
        for id, queue in enumerate(self.out_queues):
            queue: DispatcherToAgentQueue
            queue.put(self.budgets[id], self.cluster_assignments, self.token_dist)

    def recieve_data(self) -> None:
        for id, queue in enumerate(self.incoming_queues):
            queue: AgentToDispatcherQueue
            data = queue.get()
            id = data[0]
            budget = data[1]
            pref = data[2]
            self.set_bid(id, budget, pref)

            

    