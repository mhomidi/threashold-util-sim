import numpy as np
from src.scheduler import Scheduler


DIST_SAMPLE_NUMBER = 1000
DEFAULT_N = 10
DEFAULT_TOTAL_BUDGET = 100
CLUSTERS_NUMBER = 10
EPSILON = 1e-3


class Agent:
    def __init__(self, budget: int = int(DEFAULT_TOTAL_BUDGET / DEFAULT_N)) -> None:
        self.budget = budget
        self.weights = list()
        self.loss = list()
        for i in range(budget):
            self.weights.append(np.ones(DIST_SAMPLE_NUMBER))
            self.loss.append(np.zeros(DIST_SAMPLE_NUMBER))
        self.utils = np.random.rand(CLUSTERS_NUMBER).tolist()
        self.report = None
    
    def get_u_thr(self):
        dist = self.weights[self.budget - 1]
        rand = np.random.random()
        self.u_threshold = 0
        sum2uthr = 0.
        for index, p in enumerate(dist):
            sum2uthr += p / DIST_SAMPLE_NUMBER
            if sum2uthr >= rand:
                self.u_threshold = float(index) / DIST_SAMPLE_NUMBER
                break
        return self.u_threshold
    
    def update_loss(self, xs, xs_u, best_u):
        curr_loss = xs_u - best_u
        self.loss[self.budget - 1][xs] += curr_loss
    
    def update_weight(self, xs):
        l = self.loss[self.budget - 1][xs]
        self.weights[self.budget - 1][xs] *= np.exp(EPSILON * l)

    def get_preferences(self, threshold: float) -> list:
        us = self.utils.copy()
        pref = []
        l = CLUSTERS_NUMBER - 1
        while l >= 0:
            m = max(us)
            if m >= threshold:
                c_id = us.index(m)
                pref.append(c_id)
                us[c_id] = -1.
            else:
                break
            l -= 1
        return pref
            
    def train(self):
        if not self.report:
            return
        pref = self.get_preferences(0.0)
        self.report[self.id][1] = pref.copy()
        s = Scheduler()
        s.set_report(self.report)
        assignments = s.schedule()
        my_assignments = []
        for c_id, agent_id in enumerate(assignments):
            if agent_id == self.id:
                my_assignments.append(c_id)
        sorted_assigned_pref = []
        for element in pref:
            if element in my_assignments:
                sorted_assigned_pref.append(element)
        sorted_assigned_pref.reverse()
        l = len(sorted_assigned_pref)
        # TODO: need to find xs and update losses and weights
        start, end = 0, 0
        best_util = self.find_best_util(sorted_assigned_pref)
        for i in range(l + 1):
            if i == l:
                end = DIST_SAMPLE_NUMBER
            else: 
                end = int(self.utils[sorted_assigned_pref[i]] * DIST_SAMPLE_NUMBER)
                # print(start, end, self.utils[sorted_assigned_pref[i]])
            xs = [j for j in range(start, end)]
            u_xs = self.calculate_util(sorted_assigned_pref, i)
            # print(start, end, u_xs, best_util)
            self.update_loss(xs, u_xs, best_util)
            self.update_weight(xs)
            start = end
        self.report = None
            
            
    def find_best_util(self, sorted_assign) -> float:
        best_util = -10e10
        for i in range(len(sorted_assign)):
            util = self.calculate_util(sorted_assign, i)
            if util > best_util:
                best_util = util
        return best_util


        
    def calculate_util(self, sorted_assign, index) -> float:
        util = 0
        for i in range(index, len(sorted_assign)):
            util += self.utils[sorted_assign[i]]
        return util - ((len(sorted_assign) - index) * self.get_expected_util_per_token())
        # return util
    
    def get_expected_util_per_token(self):
        return sum(self.utils) / float(len(self.utils))

    def set_id(self, id: int) -> None:
        self.id = id

    def get_id(self) -> int:
        return self.id

    def set_report(self, report: list) -> None:
        self.report = report

    def get_budget(self) -> int:
        return self.budget
    
    def set_budget(self, budget: int) -> None:
        self.budget = budget
    