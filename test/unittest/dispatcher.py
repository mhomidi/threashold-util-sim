
import unittest
from utils.pipe import *
from modules.dispatcher.pref import PrefDispatcher
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

# from modules.agents import Agent
# from modules.scheduler.most_token_first import MostTokenFirstScheduler
# from modules.applications.ditribution import DistributionApplication
# from modules.policies.actor_critic import ActorCriticPolicy


class TestDispatcher(unittest.TestCase):

    def test_add_queue(self):
        q11 = Pipe(0)
        q12 = Pipe(0)

        q21 = Pipe(1)
        q22 = Pipe(1)

        dp = PrefDispatcher()

        dp.connect(q11, q12)
        assert (len(dp.incoming_pipes) == 1)
        assert (len(dp.out_pipes) == 1)

        dp.connect(q21, q22)
        assert (len(dp.incoming_pipes) == 2)
        assert (len(dp.out_pipes) == 2)

    def test_set_bid(self):
        q11 = Pipe(0)
        q12 = Pipe(0)
        q21 = Pipe(1)
        q22 = Pipe(1)
        dp = PrefDispatcher()
        dp.connect(q11, q12)
        dp.connect(q21, q22)

        dp.set_bid(0, 10, [5, 4, 3])
        dp.set_bid(1, 10, [4, 1, 1])

        assert (dp.report[0][0] == 10)
        assert (dp.report[1][0] == 10)

        assert (dp.report[0][1] == [5, 4, 3])
        assert (dp.report[1][1] == [4, 1, 1])


if __name__ == "__main__":
    unittest.main()
