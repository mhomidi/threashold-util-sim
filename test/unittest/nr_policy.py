
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from utils import constant
from config import config
from modules.agents.no_regret_policy_agent import NoRegretWithPolicyAgent
import unittest


class TestNoRegretWithPolicy(unittest.TestCase):

    def test_init(self):
        agent = NoRegretWithPolicyAgent(
            10, u_gen_type=constant.U_GEN_DISTRIBUTION)

        assert (len(agent.weights) == config.POLICY_NUM)
        assert (len(agent.loss) == config.POLICY_NUM)
        assert (agent.weights[0] == 1.)
        assert (agent.loss[0] == 0.)

    def test_agent_train(self):
        agent1 = NoRegretWithPolicyAgent(10, u_gen_type=constant.U_GEN_DISTRIBUTION)
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
