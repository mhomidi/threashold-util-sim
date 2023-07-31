
import numpy as np
from config import config
from utils import constant
from utils.distribution import UtilityGenerator, UniformMeanGenerator
from modules.markov.application import Application


class Agent:
    
    def __init__(
                self, budget: int,
                u_gen_type=constant.U_GEN_MARKOV, 
                mean_u_gen=UniformMeanGenerator(),
                application=None
                ) -> None:
        self.budget = budget
        self.u_get_type = u_gen_type
        if u_gen_type is constant.U_GEN_MARKOV and application is None:
            raise Exception()
        self.application: Application = application
        self.utils = np.random.rand(config.CLUSTERS_NUM).tolist()
        self.round_util = 0
        self.utility_generator = UtilityGenerator(mean_u_gen)

    def get_u_thr(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def get_preferences(self, threshold: float) -> list:
        us = self.utils.copy()
        pref = []
        l = config.CLUSTERS_NUM - 1
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
        if self.u_get_type is constant.U_GEN_DISTRIBUTION:
            self.utils = self.utility_generator.get_utilities()
        elif self.u_get_type is constant.U_GEN_MARKOV:
            self.application.go_next_state()
            self.utils = self.application.get_curr_state().get_val()

    def get_round_utility(self):
        util = 0.0
        for c_id, agent_id in enumerate(self.assignment):
            if agent_id == self.id:
                util += self.utils[c_id]
        return util
    
    def get_cluster_utility(self, cluster_id):
        return self.utils[cluster_id]