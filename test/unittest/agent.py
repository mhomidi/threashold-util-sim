
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from modules.agents import Agent
from modules.applications.distribution import DistributionApplication
from modules.policies.actor_critic import ActorCriticPolicy
import unittest
import numpy as np
from config import config

class TestAgent(unittest.TestCase):
    
    def test_init(self):
        app = DistributionApplication()
        policy = ActorCriticPolicy(config.BUDGET)
        agent = Agent(config.BUDGET, app, policy)
        assert(agent.application == app)
        assert(agent.policy == policy)


    def test_get_pref(self):
        app = DistributionApplication()
        policy = ActorCriticPolicy(config.BUDGET)
        agent = Agent(config.BUDGET, app, policy)

        pref = agent.get_preferences()
        u_thr = policy.u_thr_index / config.THRESHOLDS_NUM

        for cluster_id in pref:
            assert(app.curr_state.get_utils()[cluster_id] >= u_thr)
        

    def test_get_utility(self):
        app = DistributionApplication()
        policy = ActorCriticPolicy(config.BUDGET)
        agent = Agent(config.BUDGET, app, policy)
        agent.set_id(0)
        agent.get_preferences()

        assignment = np.random.randint(0, 3, config.CLUSTERS_NUM).tolist()
        u = 0.
        for cid, id in enumerate(assignment):
            if id == agent.get_id():
                u += agent.get_cluster_utility(cid)
        
        agent.set_assignment(assignment)
        assert(u == agent.get_round_utility())


if __name__ == "__main__":
    unittest.main()
