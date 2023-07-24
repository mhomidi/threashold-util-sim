

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from config.config import *
from modules.agents.actor_critic_agent import ActorCriticAgent
from modules.scheduler.round_robin import RoundRobinScheduler
from modules.dispatcher import Dispatcher
from utils.report import Report
from utils import distribution


if __name__ == "__main__":
    n = int(sys.argv[1])
    reporter = Report()
    agents = list()

    for i in range(n):
        agents.append(ActorCriticAgent(10, distribution.UniformMeanGenerator()))
        reporter.add_agent(agents[i])

    sched = RoundRobinScheduler(n)
    dp = Dispatcher()

    for i in range(n):
        dp.add_agent(agents[i])

    dp.set_scheduler(sched)

    for episode in range(int(AC_EPISODES)):
        prefs = list()

        for i in range(n):
            prefs.append(agents[i].get_preferences(0))

        for i in range(n):
            dp.set_bid(agents[i].get_id(), prefs[i])

        dp.dispatch_report()

        assignments = sched.schedule()
        dp.dispatch_assignments(assignments)
        reporter.generate_utilities_row()

        for i in range(n):
            agents[i].update_utils()
        if episode % 500 == 0:
            print("episode {e} Done".format(e=episode))
    
    reporter.write_data(UTILITY_DATA_TYPE)
        