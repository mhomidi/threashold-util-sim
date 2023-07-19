
import numpy as np
from config.config import *


CLUSTERS_NUMBER = 10

class Agent:
    
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.utils = np.random.rand(CLUSTERS_NUM).tolist()
        self.round_util = 0

    def get_u_thr(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

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

    def set_assignment(self, assignment: list) -> None:
        self.assignment = assignment

    def update_utils(self):
        self.utils = np.random.rand(CLUSTERS_NUM).tolist()

    def get_round_utility(self):
        util = 0.0
        for c_id, agent_id in enumerate(self.assignment):
            if agent_id == self.id:
                util += self.utils[c_id]
        return util