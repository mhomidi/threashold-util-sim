
import config
from modules.applications import Application
from modules.policies import Policy
from utils.queue import DispatcherToAgentQueue, AgentToDispatcherQueue
import time
import numpy as np


class Agent:
    
    def __init__(
                self, budget: int,
                application: Application,
                policy: Policy 
                ) -> None:
        self.budget = budget
        self.application: Application = application
        self.assignment = list()
        self.policy = policy
        self.incoming_queue = None
        self.out_queue = None
        self.token_dist = [0. for _ in range(config.get('token_dist_sample'))]
        self.utils_history = list()
        self.budgets_history = list()

    def get_preferences(self) -> list:
        us = self.application.get_curr_state().get_utils().copy()
        self.utils = us.copy()
        us.sort()
        data = [self.budget] + us + self.token_dist
        threshold = self.policy.get_u_thr(data)
        us = np.array(self.utils)
        arg_sort_us = np.apply_along_axis(lambda x: x, axis=0, arr=us).argsort()[::-1]

        pref = arg_sort_us[us[arg_sort_us] > threshold].tolist()
        self.application.go_next_state()
        return pref
    
    def connect(self,
                   incoming_queue: DispatcherToAgentQueue,
                   out_queue: AgentToDispatcherQueue
                   ):
        self.incoming_queue = incoming_queue
        self.out_queue = out_queue
        self.id = out_queue.get_id()

    def send_data(self):
        pref = self.get_preferences()
        self.out_queue.put(self.budget, pref)

    def recieve_data(self):
        data = self.incoming_queue.get()
        self.budget = data[1]
        self.assignment = data[2]
        self.token_dist = data[3]
    
    def train_policy(self):
        reward = self.get_round_utility()
        self.utils_history.append(reward)
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

            while self.incoming_queue.is_empty():
                time.sleep(0.0001)
            
            self.recieve_data()
            self.train_policy()
            