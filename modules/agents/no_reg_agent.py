
import random
from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.agents import Agent
from utils.distribution import UniformMeanGenerator
import math
from utils import constant
from config import config


class NoRegretAgent(Agent):
    def __init__(
        self, budget: int = config.BUDGET,
        n=config.DEFAULT_NUM_AGENT,
        u_gen_type=constant.U_GEN_MARKOV,
        mean_u_gen=UniformMeanGenerator(),
        application=None
    ) -> None:
        self.weights = list()
        self.loss = list()
        self.number_of_agent = n
        for i in range(budget * n):
            self.weights.append([1. for _ in range(config.DIST_SAMPLE)])
            self.loss.append([0. for _ in range(config.DIST_SAMPLE)])
        self.report = None
        super(NoRegretAgent, self).__init__(budget=budget,
                                            u_gen_type=u_gen_type,
                                            mean_u_gen=mean_u_gen,
                                            application=application
                                            )

    def get_u_thr(self):
        dist = self.weights[self.budget - 1]
        rand = random.random() * sum(dist)
        self.u_threshold = 0
        sum2uthr = 0.
        for index, p in enumerate(dist):
            sum2uthr += p
            if sum2uthr >= rand:
                self.u_threshold = float(index) / config.DIST_SAMPLE
                break
        return self.u_threshold

    def update_loss(self, xs, xs_u, best_u):
        curr_loss = (xs_u - best_u) / best_u
        for x in xs:
            self.loss[self.budget - 1][x] += curr_loss

    def update_weight(self, xs):
        for x in xs:
            l = self.loss[self.budget - 1][x]
            self.weights[self.budget - 1][x] = self.weights[self.budget -
                                                            1][x] * math.exp(config.EPSILON * l)

    def get_sorted_assigned_pref(self) -> list:
        if not self.report:
            return
        self.prev_budget = self.report[self.id][0]
        pref = self.get_preferences(0.0)
        self.report[self.id][1] = pref.copy()
        s = MostTokenFirstScheduler()
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
        return sorted_assigned_pref

    def train(self):
        sorted_assigned_pref = self.get_sorted_assigned_pref()
        l = len(sorted_assigned_pref)
        start, end = 0, 0
        best_util = self.find_best_util(sorted_assigned_pref)
        if best_util == -10e10:
            return
        for i in range(l + 1):
            if i == l:
                end = config.DIST_SAMPLE
                u_xs = 0.
            else:
                end = int(self.utils[sorted_assigned_pref[i]]
                          * config.DIST_SAMPLE)
                u_xs = self.calculate_util(sorted_assigned_pref, i)
            xs = [j for j in range(start, end)]
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
        spent_token = 0
        for i in range(index, len(sorted_assign)):
            util += self.utils[sorted_assign[i]]
            spent_token += 1
        return util + (self.prev_budget - spent_token) * self.application.get_mean_util()
        # return util - ((len(sorted_assign) - index) * self.get_expected_util_per_token())
        # return util

    def get_expected_util_per_token(self):
        return sum(self.utils) / float(len(self.utils))
