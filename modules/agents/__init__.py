
import config
from modules.applications import Application
from modules.applications.queue import QueueApplication
from modules.policies import Policy
from utils.pipe import *
import time
import numpy as np


class Agent:
    def __init__(
        self,
        application: Application,
        weight: float = 1. / config.get('default_agent_num')
    ) -> None:
        self.application: Application = application
        self.incoming_pipe = None
        self.out_pipe = None
        self.assignment = list()
        self.weight = weight
        self.rewards_history = list()
        self.budgets_history = list()
        self.utils_history = list()

    def connect(self,
                incoming_pipe: Pipe,
                out_pipe: Pipe
                ):
        self.incoming_pipe = incoming_pipe
        self.out_pipe = out_pipe
        self.id = out_pipe.get_id()

    def send_data(self):
        raise NotImplementedError()

    def recieve_data(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


class QueueAgent(Agent):
    def __init__(
        self,
        application: QueueApplication,
        weight: float = 1. / config.get('default_agent_num')
    ) -> None:
        super().__init__(application, weight)
        self.application: QueueApplication
        self.passed_jobs: int = 0

    def send_data(self):
        data = dict()
        data['throughput'] = self.application.get_throughput()
        data['q_length'] = self.application.get_length()
        self.out_pipe.put(data=data)

    def recieve_data(self):
        data = self.incoming_pipe.get()
        if data:
            self.passed_jobs = data['passed']
            self.assignment = data['assignments']

    def run(self):
        for _ in range(config.get('episodes')):
            self.send_data()

            while self.incoming_pipe.is_empty():
                time.sleep(0.0001)

            self.recieve_data()
            self.application.reduce_length(self.passed_jobs)
            self.rewards_history.append(self.application.get_length())
            self.passed_jobs = 0
            self.application.go_next_state()


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
