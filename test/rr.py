
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from modules.scheduler.round_robin import RoundRobinScheduler
from modules.policies.fixed_threshold import FixedThresholdPolicy
from modules.applications.markov import MarkovApplication
from modules.agents import Agent
from utils.queue import AgentToDispatcherQueue, DispatcherToAgentQueue
from config import config
from utils.report import *
import threading

json_path = os.path.dirname(os.path.abspath(__file__))


n_agent = config.DEFAULT_NUM_AGENT

def agent_send(agent: Agent):
    agent.send_data()

def agent_recieve_train(agent: Agent):
    agent.recieve_data()
    agent.train_policy()


def main():
    sched = RoundRobinScheduler(config.DEFAULT_NUM_AGENT)
    dp = sched.get_dispatcher()
    agents = list()
    reporter = Report()

    # setting up the agents
    for i in range(n_agent):
        policy = FixedThresholdPolicy()
        app = MarkovApplication()
        app.init_from_json(json_file=json_path + "/json/agents/base.json")
        agent = Agent(config.BUDGET, app, policy)

        a2d_q = AgentToDispatcherQueue(i)
        d2a_q = DispatcherToAgentQueue(i)

        dp.connect(a2d_q, d2a_q)
        agent.connect(d2a_q, a2d_q)
        agents.append(agent)
        reporter.add_agent(agent)

    # Running the episodes  
    for episode in range(config.AC_EPISODES):
        send_threads = list()
        for agent in agents:
            # agent_send(agent)
            t = threading.Thread(target=agent_send, args=(agent,))
            t.start()
            send_threads.append(t)
        for t in send_threads:
            t.join()
        
        reporter.generate_tokens_row()
        reporter.generate_token_distributions_row()
        dp.recieve_data()
        
        if dp.get_report() is None:
            continue

        sched.set_report(dp.get_report())
        sched.schedule()
        sched.dist_tokens()
        dp.send_data()

        train_threads = list()
        for agent in agents:
            # agent_recieve_train(agent)
            t = threading.Thread(target=agent_recieve_train, args=(agent,))
            t.start()
            train_threads.append(t)

        for t in train_threads:
            t.join()
        
        reporter.generate_utilities_row()
        
        if episode % 500 == 499:
            print("episode {e} done".format(e=episode + 1))

    reporter.write_data(UTILITY_DATA_TYPE)
    reporter.write_data(TOKEN_DATA_TYPE)
    reporter.write_data(TOKEN_DIST_TYPE)


if __name__ == "__main__":
    main()
    






# def main(n: int, policy_type: str, sched_type:str, app_type: str):
#     sched_class = SCHEDULERS[sched_type]
#     policy_class = POLICIES[policy_type]
#     app_class = APPLICATIONS[app_type]

#     for i in range(n):
#         policy = policy_class(config.BUDGET)
#         app = app_class()
