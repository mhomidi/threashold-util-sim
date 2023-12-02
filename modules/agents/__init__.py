import config
from modules.applications import Application
from modules.applications.queue import QueueApplication
from modules.policies import Policy
from utils.pipe import *
import time
import numpy as np


class Agent:
    def __init__(self, agent_id, applications, weight):
        self.agent_id = agent_id
        self.applications = applications
        self.weight = weight
        self.cluster_size = config.get(config.CLUSTER_NUM)
        assert(self.cluster_size == len(applications))
        self.assignments = np.zeros(self.cluster_size)
        self.rewards = np.zeros(self.cluster_size)
        self.utilities = np.zeros(self.cluster_size)
        self.assignments_history = list()
        self.rewards_history = list()
        self.utilities_history = list()

    def run_agent(self, iteration, assignments):
        self.assignments_history.append(self.assignments)
        self.rewards_history.append(self.rewards)
        self.utilities_history.append(self.utilities)


class QueueAgent(Agent):
    def __init__(self, agent_id, applications, weight, arrival_generator, load_calculator, load_balancer):
        super().__init__(agent_id, applications, weight)
        self.arrival_generator = arrival_generator
        self.queue_lengths = np.zeros(self.cluster_size)
        self.departure_rates = np.zeros(self.cluster_size)
        self.load = np.zeros(self.cluster_size)
        self.load_history = list()
        self.load_calculator = load_calculator
        self.load_balancer = load_balancer

    def run_agent(self, iteration, assignments):
        super().run_agent(iteration, assignments)
        self.load_history.append(self.load)

        self.assignments = assignments
        self.rewards = self.utilities * self.assignments

        arrivals = self.arrival_generator.generate()
        per_queue_arrivals = self.load_balancer.balance_load(arrivals, self.load)
        for i, app in enumerate(self.applications):

            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.go_next_state()

            self.queue_lengths[i] = app.get_current_queue_length()
            self.departure_rates[i] = app.get_departure_rate()

            # new_utility = app.get_utility()
            # self.rewards[i] = self.utilities[i] - new_utility
            # self.utilities[i] = new_utility
            self.utilities[i] = app.get_utility()

        self.load = self.load_calculator.calculate_load(self.queue_lengths, self.departure_rates)
        return self.utilities


class PrefAgent(Agent):

    def __init__(
        self, budget: int,
        application: Application,
        policy: Policy,
        weight: float = 1. / config.get('default_agent_num')
    ) -> None:
        super().__init__(application, weight)
        self.budget = budget
        self.policy = policy
        self.token_dist = [0. for _ in range(config.get('token_dist_sample'))]

    def get_preferences(self) -> list:
        us = self.application.get_curr_state().get_utils().copy()
        self.utils = us.copy()
        self.utils_history.append(self.utils)
        us.sort()
        data = [self.budget] + us + self.token_dist
        threshold = self.policy.get_u_thr(data)
        us = np.array(self.utils)
        arg_sort_us = np.apply_along_axis(
            lambda x: x, axis=0, arr=us).argsort()[::-1]
        pref = arg_sort_us[us[arg_sort_us] >= threshold].tolist()
        self.application.go_next_state()
        return pref

    def send_data(self):
        pref = self.get_preferences()
        data = dict()
        data['budget'] = self.budget
        data['pref'] = pref
        self.out_pipe.put(data=data)

    def recieve_data(self):
        data = self.incoming_pipe.get()
        if data:
            self.budget = data['budget']
            self.assignment = data['assignments']
            self.token_dist = data['token_dist']

    def train_policy(self):
        reward = self.get_round_utility()
        self.rewards_history.append(reward)
        us = self.application.get_curr_state().get_utils().copy()
        us.sort()
        next_state_data = [self.budget] + us + self.token_dist
        self.policy.train(reward, next_state_data)

    def get_round_utility(self):
        us = np.array(self.utils)
        return us[np.array(self.assignment) == self.id].sum()

    def get_cluster_utility(self, cluster_id):
        return self.utils[cluster_id]

    def run(self):
        for _ in range(config.get('episodes')):
            self.send_data()

            while self.incoming_pipe.is_empty():
                time.sleep(0.0001)

            self.recieve_data()
            self.train_policy()
