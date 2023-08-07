
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import unittest
import config as config
from modules.policies.actor_critic import ActorCriticPolicy


class TestActorCriticPolicy(unittest.TestCase):

    def test_get_u_thr(self):
        policy = ActorCriticPolicy(10)
        input_data = [0.2 for i in range(config.get('cluster_num'))]
        input_data = [10] + input_data

        u_thr = policy.get_u_thr(input_data)

        assert(0. <= u_thr <= 1.)


    def test_train(self):
        policy = ActorCriticPolicy(10)
        input_data = [0.2 for _ in range(config.get('cluster_num'))]
        input_data = [10] + input_data
        policy.get_u_thr(input_data)

        input_data = [0.3 for _ in range(config.get('cluster_num'))]
        input_data = [7] + input_data
        policy.train(1.5, input_data)



if __name__ == "__main__":
    unittest.main()