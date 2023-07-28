import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from modules.agents.no_reg_agent import NoRegretAgent
from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.dispatcher import Dispatcher
from utils import distribution
from utils.report import NRAgentReporter, UTILITY_DATA_TYPE, TOKEN_DATA_TYPE
from config import config


if __name__ == "__main__":
    n = int(sys.argv[1])
    reporter = NRAgentReporter()
    agents = list()

    for i in range(n):
        agents.append(NoRegretAgent(
            budget=10,
            n=n,
            u_gen_type=config.U_GEN_DISTRIBUTION,
            mean_u_gen=distribution.PoissonMeanGenerator()
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
    for i in range(n):
        reporter.write_weights(agents[i])
