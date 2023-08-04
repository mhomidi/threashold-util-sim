import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from modules.scheduler.most_token_first import MostTokenFirstScheduler
from modules.policies.actor_critic import ActorCriticPolicy
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
    sched = MostTokenFirstScheduler()
    dp = sched.get_dispatcher()
    agents = list()
    threads = list()

    # setting up the agents
    for i in range(n_agent):
        policy = ActorCriticPolicy(config.BUDGET)
        app = MarkovApplication()
        app.init_from_json(json_file=json_path + "/json/agents/base.json")
        agent = Agent(config.BUDGET, app, policy)

        a2d_q = AgentToDispatcherQueue(i)
        d2a_q = DispatcherToAgentQueue(i)

        dp.connect(a2d_q, d2a_q)
        agent.connect(d2a_q, a2d_q)
        agents.append(agent)
        t = threading.Thread(target=agent.start)
        t.start()
        threads.append(t)
    
    t = threading.Thread(target=sched.start)
    t.start()
    threads.append(t)
    
    for t in threads:
        t: threading.Thread
        t.join()


if __name__ == "__main__":
    main()