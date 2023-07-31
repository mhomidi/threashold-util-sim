

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from config import config
from modules.agents.actor_critic_agent import ActorCriticAgent
from modules.scheduler.round_robin import RoundRobinScheduler
from modules.dispatcher import Dispatcher
from utils.report import *
from utils import constant
from modules.markov import application



if __name__ == "__main__":
    n = int(sys.argv[1])
    reporter = Report()
    agents = list()

    for i in range(n):
        agent_config_json = os.path.dirname(os.path.abspath(__file__)) + "/../json/agents/a" + str(i) + "_conf.json"
        app = application.Application()
        app.init_from_json(json_file=agent_config_json)

        agents.append(ActorCriticAgent(
            budget=10, 
            u_gen_type=constant.U_GEN_MARKOV,
            application=app
            ))
        reporter.add_agent(agents[i])


    sched = RoundRobinScheduler(n)
    dp = Dispatcher()

    for i in range(n):
        dp.add_agent(agents[i])

    dp.set_scheduler(sched)

    for episode in range(int(config.AC_EPISODES)):
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
        