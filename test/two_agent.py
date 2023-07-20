import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from modules.agents.no_reg_agent import NoRegretAgent
from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.dispatcher import Dispatcher
from utils.report import NRAgentReporter

MAX_COUNT = 1e4

if __name__ == "__main__":
    a1 = NoRegretAgent(10, 2)
    a2 = NoRegretAgent(10, 2)
    dp = Dispatcher()
    sched = MostTokenFirstScheduler()

    dp.add_agent(a1)
    dp.add_agent(a2)
    dp.set_scheduler(sched)

    count = 0

    while count < MAX_COUNT:
        a1_pref = a1.get_preferences(a1.get_u_thr())
        a2_pref = a2.get_preferences(a2.get_u_thr())

        dp.set_bid(a1.get_id(), a1_pref)
        dp.set_bid(a2.get_id(), a2_pref)

        dp.dispatch_report()

        sched.schedule()

        a1.train()
        a2.train()

        sched.dist_tokens()

        dp.update_budgets(sched.get_new_budgets())
        
        count += 1

    reporter = NRAgentReporter()
    reporter.write_weights(a1)
    reporter.write_weights(a2)
    