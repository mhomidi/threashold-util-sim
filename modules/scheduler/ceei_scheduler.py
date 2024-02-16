from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np


class CEEIScheduler(Scheduler):
    def __init__(self, agent_weights, num_agents, num_nodes):
        super().__init__(agent_weights, num_agents, num_nodes)

    def run_scheduler(self, iteration, demands):
        x = cp.Variable(shape=(self.num_agents, self.num_nodes), nonneg=True)
        v = cp.Variable(shape=self.num_agents)
        u = cp.Parameter(shape=(self.num_agents, self.num_nodes), value=demands)
        constraints = [
            cp.sum(x, axis=0) <= np.ones(self.num_nodes),
            v <= cp.sum(cp.multiply(x, u), axis=1)
        ]
        objective = cp.Maximize(cp.sum(cp.log(v) @ self.agent_weights))
        prob = cp.Problem(objective, constraints)
        prob.solve()
        if prob.status != cp.OPTIMAL:
            raise Exception(prob.status)
        
        allocation = self.get_alloc(x.value)
        return allocation, None
