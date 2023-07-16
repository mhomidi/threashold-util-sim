
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


from src.agent import *
import unittest
import matplotlib.pyplot as plt

class TestAgent(unittest.TestCase):

    def test_agent_init(self):
        agent1 = Agent(budget=10)
        agent1.set_id(0)
        assert(len(agent1.weights) == agent1.budget)
        assert(len(agent1.loss) == agent1.budget)
        assert(len(agent1.utils) == CLUSTERS_NUMBER)

    def test_agent_pref(self):
        agent1 = Agent()
        agent1.set_id(0)
        
        u_thr = agent1.get_u_thr()
        assert(u_thr <= 1.)
        assert(u_thr >= 0.)

        agent_pref = agent1.get_preferences(u_thr)
        true_pref = []
        l = len(agent1.utils)
        us = agent1.utils.copy()
        while l > 0:
            u = max(us)
            if u >= u_thr:
                c_id = us.index(u)
                true_pref.append(c_id)
                us[c_id] = -1.
            else:
                break
            l -= 1
        assert(true_pref == agent_pref)

    def test_agent_train(self):
        agent1 = Agent(budget=10)
        agent1.set_id(0)
        pref = agent1.get_preferences(agent1.get_u_thr())
        report = [
            [10, pref],
            [10, [1, 3, 4, 5, 6]]
        ]

        agent1.set_report(report)
        agent1.train()


if __name__ == "__main__":
    unittest.main()
