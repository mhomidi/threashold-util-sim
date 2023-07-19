

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")


from modules.agents.actor_critic_agent import ActorCriticAgent
from modules.scheduler import Scheduler
from modules.dispatcher import Dispatcher

EPISODES = 10e2


if __name__ == "__main__":
    n = int(sys.argv[1])
    agents = list()

    for i in range(n):
        agents.append(ActorCriticAgent(10))

    sched = Scheduler()
    dp = Dispatcher()

    for i in range(n):
        dp.add_agent(agents[i])

    dp.set_scheduler(sched)

    for episode in range(1000):
        prefs = list()

        for i in range(n):
            prefs.append(agents[i].get_preferences(agents[i].get_u_thr()))

        for i in range(n):
            dp.set_bid(agents[i].get_id(), prefs[i])

        dp.dispatch_report()

        assignments = sched.schedule()

        sched.dist_tokens()
        dp.update_budgets(sched.get_new_budgets())

        for i in range(n):
            agents[i].set_assignment(assignments)
            agents[i].train()
            agents[i].update_utils()
        