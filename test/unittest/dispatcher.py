
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from modules.dispatcher import Dispatcher
# from modules.agents import Agent
# from modules.scheduler.most_token_first import MostTokenFirstScheduler
# from modules.applications.ditribution import DistributionApplication
# from modules.policies.actor_critic import ActorCriticPolicy
from utils.queue import *

from config import config


import unittest

class TestDispatcher(unittest.TestCase):

    def test_add_queue(self):
        q11 = AgentToDispatcherQueue(0)
        q12 = DispatcherToAgentQueue(0)

        q21 = AgentToDispatcherQueue(1)
        q22 = DispatcherToAgentQueue(1)

        dp = Dispatcher()

        dp.connect(q11, q12)
        assert(len(dp.incoming_queues) == 1)
        assert(len(dp.out_queues) == 1)

        dp.connect(q21, q22)
        assert(len(dp.incoming_queues) == 2)
        assert(len(dp.out_queues) == 2)

    def test_set_bid(self):
        q11 = AgentToDispatcherQueue(0)
        q12 = DispatcherToAgentQueue(0)
        q21 = AgentToDispatcherQueue(1)
        q22 = DispatcherToAgentQueue(1)
        dp = Dispatcher()
        dp.connect(q11, q12)
        dp.connect(q21, q22)

        dp.set_bid(0, 10, [5, 4, 3])
        dp.set_bid(1, 10, [4, 1, 1])

        assert(dp.report[0][0] == 10)
        assert(dp.report[1][0] == 10)

        assert(dp.report[0][1] == [5, 4, 3])
        assert(dp.report[1][1] == [4, 1, 1])


if __name__ == "__main__":
    unittest.main()