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
    def __init__(self, agents, token_coefficient=100):
        super().__init__(agents)
        self.tokens = np.zeros(self.num_agents)
        for i in range(0, self.num_agents):
            self.tokens[i] = self.agents[i].get_weight() * token_coefficient

    def run_scheduler(self, iteration, demands: np.ndarray):
        # demands[i] = np.array([0, 3, 1, 2, 0, 0, 0, 0, 0])
        # TODO: bring the logic here! return new assignments and tokens once calculated
        assignment = [-1 for _ in range(self.num_clusters)]
        gathered_token = 0
        for _ in range(self.num_clusters):
            max_budget_agent_index = self.find_max_budget_agent_index()
            if max_budget_agent_index == -1:
                self.assgin_randomly_available_clusters(assignment)
                break
            cluster_idx = np.argmax(demands[max_budget_agent_index])
            assignment[cluster_idx] = max_budget_agent_index
            self.tokens[max_budget_agent_index] -= 1
            gathered_token += 1
            demands[:, cluster_idx] = 0
        self.create_one_hot_assignments(assignment)
        self.redistribute_tokens(gathered_token)
        return self.assignments, self.tokens

    def create_one_hot_assignments(self, assignments: list):
        self.assignments = np.zeros((self.num_agents, self.num_clusters))
        for c, a in enumerate(assignments):
            self.assignments[a, c] = 1

    def find_max_budget_agent_index(self, demands: np.ndarray) -> int:
        max_budget_agent_indices = np.argsort(self.tokens).tolist()[::-1]
        for i in max_budget_agent_indices:
            if demands[i].sum() >= 0 and self.tokens[i] >= 1.:
                return i
        return -1

    def assgin_randomly_available_clusters(self, assignment: list) -> int:
        weights = np.array([agent.get_weight() for agent in self.agents])
        p = weights / weights.sum()
        for i in assignment:
            if i == -1:
                r = np.random.choice([i for i in range(self.num_agents)], p=p)
                assignment[i] = r

    def redistribute_tokens(self, gathered_tokens: int):
        weights = np.array([agent.get_weight() for agent in self.agents])
        weights /= weights.sum()
        self.tokens += weights * gathered_tokens


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
