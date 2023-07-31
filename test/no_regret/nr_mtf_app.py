

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


from config import config
from modules.agents.no_regret_policy_agent import NoRegretWithPolicyAgent
from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.dispatcher import Dispatcher
from utils.report import *
from modules.markov import application
from utils import constant



if __name__ == "__main__":
    n = int(sys.argv[1])
    reporter = NRAgentReporter()
    agents = list()

    for i in range(n):
        agent_config_json = os.path.dirname(os.path.abspath(__file__)) + "/../json/agents/a" + str(i) + "_conf.json"
        app = application.Application()
        app.init_from_json(json_file=agent_config_json)

        agents.append(NoRegretWithPolicyAgent(
            budget=10, 
            n=n,
            u_gen_type=constant.U_GEN_MARKOV,
            application=app
            ))
        reporter.add_agent(agents[i])
    
    sched = MostTokenFirstScheduler()
    dp = Dispatcher()

    for i in range(n):
        dp.add_agent(agents[i])

    dp.set_scheduler(sched)

    for episode in range(int(config.NR_ROUNDS)):
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
        reporter.generate_utilities_row()

        for i in range(n):
            agents[i].train()
            agents[i].update_utils()
        if episode % 500 == 0:
            print("episode {e} Done".format(e=episode))
    
    reporter.write_data(UTILITY_DATA_TYPE)
    reporter.write_data(TOKEN_DATA_TYPE)
    # for i in range(n):
    #     reporter.write_weights(agents[i])
