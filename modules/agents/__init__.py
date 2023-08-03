
from config import config
from modules.applications import Application
from modules.policies import Policy
from utils.queue import DispatcherToAgentQueue, AgentToDispatcherQueue


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

    def get_preferences(self) -> list:
        us = self.application.get_curr_state().get_utils().copy()
        self.utils = us.copy()
        us.sort()
        data = [self.budget] + us
        threshold = self.policy.get_u_thr(data)
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
        self.set_budget(data[1])
        self.set_assignment(data[2])
    
    def train_policy(self):
        reward = self.get_round_utility()
        us = self.application.get_curr_state().get_utils().copy()
        us.sort()
        next_state_data = [self.budget] + us
        self.policy.train(reward, next_state_data)


    def set_id(self, id: int) -> None:
        self.id = id

    def get_id(self) -> int:
        return self.id

    def get_budget(self) -> int:
        return self.budget
    
    def set_budget(self, budget: int) -> None:
        self.budget = budget

    def set_assignment(self, assignment: list) -> None:
        self.assignment = assignment

    def get_round_utility(self):
        util = 0.0
        for c_id, agent_id in enumerate(self.assignment):
            if agent_id == self.id:
                util += self.utils[c_id]
        return util
    
    def get_cluster_utility(self, cluster_id):
        return self.utils[cluster_id]