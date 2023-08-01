
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from modules.dispatcher import *
from modules.agents import Agent
from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.applications.ditribution import DistributionApplication
from modules.policies.actor_critic import ActorCriticPolicy
from config import config


import unittest

class TestDispatcher(unittest.TestCase):

    def test_add_agent(self):
        app1 = DistributionApplication()
        policy1 = ActorCriticPolicy(config.BUDGET)
        a1 = Agent(config.BUDGET, app1, policy1)

        app2 = DistributionApplication()
        policy2 = ActorCriticPolicy(config.BUDGET)
        a2 = Agent(config.BUDGET, app2, policy2)

        dp = Dispatcher()
        sched = MostTokenFirstScheduler()
        dp.set_scheduler(sched)

        dp.add_agent(a1)
        assert(len(dp.agents) == 1)

        dp.add_agent(a2)
        assert(len(dp.agents) == 2)

    def test_set_bid(self):
        app1 = DistributionApplication()
        policy1 = ActorCriticPolicy(config.BUDGET)
        agent1 = Agent(config.BUDGET, app1, policy1)

        app2 = DistributionApplication()
        policy2 = ActorCriticPolicy(config.BUDGET)
        agent2 = Agent(config.BUDGET, app2, policy2)
        dp = Dispatcher()
        sched = MostTokenFirstScheduler()
        dp.set_scheduler(sched)
        dp.add_agent(agent1)
        dp.add_agent(agent2)

        a1_pref = agent1.get_preferences()
        a2_pref = agent2.get_preferences()
        dp.set_bid(agent1.get_id(), a1_pref)
        dp.set_bid(agent2.get_id(), a2_pref)

        report = dp.get_report()
        assert(report[0][0] == agent1.get_budget())
        assert(report[1][0] == agent2.get_budget())
        assert(report[0][1] == a1_pref)
        assert(report[1][1] == a2_pref)

    def test_dispatch_report(self):
        app1 = DistributionApplication()
        policy1 = ActorCriticPolicy(config.BUDGET)
        agent1 = Agent(config.BUDGET, app1, policy1)

        app2 = DistributionApplication()
        policy2 = ActorCriticPolicy(config.BUDGET)
        agent2 = Agent(config.BUDGET, app2, policy2)

        dp = Dispatcher()
        sched = MostTokenFirstScheduler()
        dp.set_scheduler(sched)
        dp.add_agent(agent1)
        dp.add_agent(agent2)

        a1_pref = agent1.get_preferences()
        a2_pref = agent2.get_preferences()
        dp.set_bid(agent1.get_id(), a1_pref)
        
        dp.dispatch_report()
        assert(sched.report == None)

        dp.set_bid(agent2.get_id(), a2_pref)

        dp.dispatch_report()
        report = dp.get_report()
        assert(sched.report == report)

    def test_update_budget(self):
        app1 = DistributionApplication()
        policy1 = ActorCriticPolicy(config.BUDGET)
        agent1 = Agent(config.BUDGET, app1, policy1)

        app2 = DistributionApplication()
        policy2 = ActorCriticPolicy(config.BUDGET)
        agent2 = Agent(config.BUDGET, app2, policy2)

        dp = Dispatcher()
        sched = MostTokenFirstScheduler()
        dp.set_scheduler(sched)
        dp.add_agent(agent1)
        dp.add_agent(agent2)

        a1_pref = agent1.get_preferences()
        a2_pref = agent2.get_preferences()
        dp.set_bid(agent1.get_id(), a1_pref)
        dp.set_bid(agent2.get_id(), a2_pref)

        sched.set_report(dp.get_report())
        sched.schedule()
        new_b = sched.get_new_budgets()
        dp.update_budgets(new_b)

        assert(agent1.get_budget() == new_b[0])
        assert(agent2.get_budget() == new_b[1])


if __name__ == "__main__":
    unittest.main()