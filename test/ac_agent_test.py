

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


from config.config import *
from modules.agents.actor_critic_agent import ActorCriticAgent
from modules.scheduler.MTFscheduler import MostTokenFirstScheduler
from modules.dispatcher import Dispatcher
from utils.report import Report

EPISODES = 10e2


if __name__ == "__main__":
    n = int(sys.argv[1])
    reporter = Report()
    agents = list()

    for i in range(n):
        agents.append(ActorCriticAgent(10))
        reporter.add_agent(agents[i])

    sched = MostTokenFirstScheduler()
    dp = Dispatcher()

    for i in range(n):
        dp.add_agent(agents[i])

    dp.set_scheduler(sched)

    for episode in range(1000):
        reporter.generate_tokens_row()
        prefs = list()

        for i in range(n):
            prefs.append(agents[i].get_preferences(agents[i].get_u_thr()))

        for i in range(n):
            dp.set_bid(agents[i].get_id(), prefs[i])

        dp.dispatch_report()

        assignments = sched.schedule()
        dp.dispatch_assignments(assignments)

        sched.dist_tokens()
        dp.update_budgets(sched.get_new_budgets())

        for i in range(n):
            agents[i].train()
            agents[i].update_utils()
        reporter.generate_utilities_row()
    reporter.write_data(UTILITY_DATA_TYPE)
    reporter.write_data(TOKEN_DATA_TYPE)
        