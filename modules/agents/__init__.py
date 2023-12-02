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
        self.utility = 0
        self.assignments_history = list()
        self.utility_history = list()

    def update_agent(self, iteration, assignments):
        self.assignments_history.append(self.assignments)
        self.utility_history.append(self.utility)


class QueueAgent(Agent):
    def __init__(self, agent_id, applications, weight, arrival_generator,
                 load_calculator, load_balancer):
        super().__init__(agent_id, applications, weight)
        self.applications: QueueApplication
        self.arrival_generator = arrival_generator
        self.loads = np.zeros(self.cluster_size)
        self.loads_history = list()
        self.load_calculator = load_calculator
        self.load_balancer = load_balancer

    def update_agent(self, iteration, assignments):
        super().update_agent(iteration, assignments)
        self.loads_history.append(self.loads)

        self.assignments = assignments
        arrivals = self.arrival_generator.generate()
        per_queue_arrivals = self.load_balancer.balance_load(arrivals, self.loads)
        queue_lengths = np.zeros(self.cluster_size)
        avg_departure_rates = np.zeros(self.cluster_size)

        self.utility = 0
        for i, app in enumerate(self.applications):

            app.set_arrival(per_queue_arrivals[i])
            app.set_assignment(self.assignments[i])
            app.update_state()

            queue_lengths[i] = app.get_current_queue_length()
            avg_departure_rates[i] = app.get_avg_throughput()
            self.utility += app.get_imm_throughput()

        self.loads = self.load_calculator.calculate_load(queue_lengths, avg_departure_rates)


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
        self.application.update_state()
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
