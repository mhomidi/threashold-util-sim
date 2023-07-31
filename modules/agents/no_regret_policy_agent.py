

from modules.agents.no_reg_agent import NoRegretAgent
from utils import constant
from utils.distribution import UniformMeanGenerator
from config import config
from utils import constant
from modules.policies.budget_threshold_policy import BudgetThresholdPolicy
import random
import math


class NoRegretWithPolicyAgent(NoRegretAgent):

    def __init__(self, budget: int,
                 n=config.DEFAULT_NUM_AGENT,
                 u_gen_type=constant.U_GEN_MARKOV,
                 mean_u_gen=UniformMeanGenerator(),
                 application=None) -> None:
        super().__init__(budget, n, u_gen_type, mean_u_gen, application)
        self.weights.clear()
        self.loss.clear()
        self.weights = [1. for _ in range(config.POLICY_NUM)]
        self.loss = [0. for _ in range(config.POLICY_NUM)]

    def update_loss(self, xs, xs_u, best_u):
        curr_loss = (xs_u - best_u) / best_u
        for x in xs:
            self.loss[x] += curr_loss

    def update_weight(self, xs):
        for x in xs:
            l = self.loss[x]
            self.weights[x] = self.weights[x] * \
                math.exp(config.EPSILON * l)

    def train(self):
        sorted_assigned_pref = self.get_sorted_assigned_pref()
        l = len(sorted_assigned_pref)
        best_util = self.find_best_util(sorted_assigned_pref)
        if best_util == -10e10:
            return
        for i in range(l):
            util = self.calculate_util(sorted_assigned_pref, i)
            if i > 0:
                policy_indices = self.get_policy_indices(
                    sorted_assigned_pref[i], sorted_assigned_pref[i-1])
            else:
                policy_indices = self.get_policy_indices(
                    sorted_assigned_pref[i])
            self.update_loss(policy_indices, util, best_util)
            self.update_weight(policy_indices)

    def get_policy_indices(self, leq_index: int, geq_index: int = -1) -> list:
        indices = list()
        budget_index = self.get_budget_index(self.prev_budget)
        for index in range(config.POLICY_NUM):
            policy = BudgetThresholdPolicy(index=index)
            thresholds = policy.get_thresholds()
            if (thresholds[budget_index] <= self.utils[leq_index]) and (geq_index == -1 or thresholds[budget_index] >= self.utils[geq_index]):
                indices.append(index)
        return indices

    def get_budget_index(self, budget):
        if budget < config.BUDGET - 3:
            return 0
        if budget < config.BUDGET:
            return 1
        if budget <= config.BUDGET + 3:
            return 2
        return 3
    
    def get_u_thr(self):
        rand = random.random() * sum(self.weights)
        self.u_threshold = 0
        sum2uthr = 0.
        for index, p in enumerate(self.weights):
            sum2uthr += p
            if sum2uthr >= rand:
                self.u_threshold = float(index) / config.DIST_SAMPLE
                break
        return self.u_threshold
