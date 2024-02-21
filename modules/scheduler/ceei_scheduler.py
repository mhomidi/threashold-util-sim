from modules.scheduler import Scheduler
import cvxpy as cp
import numpy as np


class CEEIScheduler(Scheduler):

    def run_scheduler(self, iteration, demands):
        max_demands = np.max(demands, axis=1)
        active_users = np.where(max_demands > 0)[0]
        num_active_users = len(active_users)
        fractional_allocations = np.zeros((self.num_agents, self.num_nodes))
        if num_active_users > 0:
            normalized_demands = (demands[active_users].T / max_demands[active_users]).T
            x = cp.Variable(shape=(num_active_users, self.num_nodes), nonneg=True)
            v = cp.Variable(shape=num_active_users)
            u = cp.Parameter(shape=(num_active_users, self.num_nodes), value=normalized_demands)
            constraints = [
                cp.sum(x, axis=0) <= np.ones(self.num_nodes),
                v <= cp.sum(cp.multiply(x, u), axis=1)
            ]
            objective = cp.Maximize(cp.sum(cp.log(v) @ self.agent_weights[active_users]))
            prob = cp.Problem(objective, constraints)
            # prob.solve(cp.SCS)
            prob.solve()
            if prob.status != cp.OPTIMAL:
                raise Exception(prob.status)
            fractional_allocations[active_users, :] = x.value
            allocation = self.get_alloc(fractional_allocations)
        else:
            allocation = fractional_allocations
        return allocation, None
