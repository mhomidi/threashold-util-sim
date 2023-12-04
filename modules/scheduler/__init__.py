import numpy as np
import config


class Scheduler:
    def __init__(self, agents):
        self.agents = agents
        self.num_agents = config.get(config.AGENT_NUM)
        self.num_clusters = config.get(config.CLUSTER_NUM)
        self.assignments = np.zeros((self.num_agents, self.num_clusters))

    def run_scheduler(self, iteration, demands):
        raise NotImplementedError


class MostTokensFirstScheduler(Scheduler):
    def __init__(self, agents, token_coefficient=1):
        super().__init__(agents)
        self.tokens = np.zeros(self.num_agents)
        for i in range(0, self.num_agents):
            self.tokens[i] = self.agents[i].get_weight() * token_coefficient

    def run_scheduler(self, iteration, demands):
        # demands[i] = np.array([0, 3, 1, 2, 0, 0, 0, 0, 0])
        # TODO: bring the logic here! return new assignments and tokens once calculated
        return self.assignments, self.tokens


class GandivaFairScheduler(Scheduler):

    class StrideScheduler:
        def __init__(self, weights):
            self.num_agents = len(weights)
            self.weights = weights
            self.passes = np.zeros(self.num_agents)
            self.strides = 100000 / self.weights

        def schedule(self, demands):
            agents_w_demands = np.arange(0, self.num_agents)[demands > 0]
            agent = -1
            if len(agents_w_demands) > 0:
                active_passes = self.passes[agents_w_demands]
                agent = agents_w_demands[np.argmin(active_passes)]
                self.passes[agent] += self.strides[agent]
            return agent

    def __init__(self, agents):
        super().__init__(agents)
        # TODO: have a list of stride schedulers

    def run_scheduler(self, iteration, demands):
        # demands[i] = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0])
        # TODO: bring the logic here! return new assignments
        return self.assignments





#
#
# class Scheduler:
#
#     def __init__(self) -> None:
#         self.report = None
#         self.cluster_assignment = [
#             -1 for _ in range(config.get('cluster_num'))]
#         self.dispatcher = None
#         self.assignment_history = list()
#
#     def set_report(self, report: list) -> None:
#         self.report = report
#
#     def get_report(self) -> list:
#         return self.report
#
#     def get_dispatcher(self) -> Dispatcher:
#         return self.dispatcher
#
#     def get_cluster_assignments(self):
#         return self.cluster_assignment
#
#     def schedule(self) -> list:
#         raise NotImplementedError
#
#     def run(self):
#         raise NotImplementedError
