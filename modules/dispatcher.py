

from modules.scheduler import Scheduler
from modules.agents import Agent
import copy


class Dispatcher:
    def __init__(self) -> None:
        self.agents = list()
        self.sched = None
        self.report = list()
        self.report_update = list()
        self.last_id = 0

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)
        self.report.append(None)
        self.report_update.append(False)
        agent.set_id(self.last_id)
        self.last_id += 1

    def dispatch_report(self):
        if not self.is_report_new():
            return
        self.sched.set_report(copy.deepcopy(self.report))
    
    def is_report_new(self):
        for i in range(self.last_id):
            if not self.report_update[i]:
                return False
        return True

    def set_bid(self, agent_id: int, bid: list) -> None:
        self.report[agent_id] = [self.agents[agent_id].get_budget(), bid]
        self.report_update[agent_id] = True

    def get_report(self) -> list:
        return self.report
    
    def set_scheduler(self, sched: Scheduler) -> None:
        self.sched = sched
    
    def update_budgets(self, budgets) -> None:
        for id, agent in enumerate(self.agents):
            agent.set_budget(budgets[id])

    def dispatch_assignments(self, assignments: list) -> None:
        for agent in self.agents:
            agent.set_assignment(assignments)

    def get_cluster_util_from_agent(self, cluster_id: int, agent_id: int) -> float:
        return self.agents[agent_id].get_cluster_utility(cluster_id)

    