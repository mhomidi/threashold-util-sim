import random
from utils.constructors import *
import config
import threading
from utils.report import *
from utils.pipe import Pipe, Pipe
from modules.agents import PrefAgent
import sys
import getopt
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


random.seed("hadi")

json_path = os.path.dirname(os.path.abspath(__file__))
markov_json_path = json_path + "/../json/agents/app0.json"


def parse_args(argv):
    data = {
        'sched': 'mtf',
        'policy': 'ac',
        'app': 'markov',
        'n': int(config.get('default_agent_num')),
        'tokens': int(config.get('budget')),
    }

    opts, _ = getopt.getopt(argv, "h:s:p:a:n:t:l:r:g", [
                            "scheduler=", "policy=", "app=", "nagent=", "tokens=", "lambda=", "thr=", "agent="])
    for opt, arg in opts:
        if opt == '-h':
            print(
                'test.py -s <scheduler> -p <policy> -a <app_type> -n <n_agnet> -t <tokens> -l <lambda> -r <thr> -g <agent>')
            sys.exit()
        elif opt in ("-s", "--sched"):
            data['sched'] = arg
        elif opt in ('-p', '--policy'):
            data['policy'] = arg
        elif opt in ('-a', '--app'):
            data['app'] = arg
        elif opt in ('-n', '--nagent'):
            data['n'] = arg
        elif opt in ('-t', '--tokens'):
            data['tokens'] = arg
        elif opt in ('-l', '--lambda'):
            data['lambda'] = arg
        elif opt in ('-r', '--thr'):
            data['thr'] = arg
        elif opt in ('-g', '--agent'):
            data['agent'] = arg

    return data


def main(data: dict):
    sched = get_scheduler(type=data['sched'])
    dp = sched.get_dispatcher()
    agents = list()
    threads = list()
    reporter = Report()
    reporter.set_scheduler(sched)

    # setting up the agents
    for i in range(data['n']):
        # get_policy should have type, budget, thr
        policy = get_policy(type=data['policy'], budget=data['tokens'])
        # get_application should have app, markov_json_path, p_lambda
        app = get_application(
            type=data['app'], markov_json_path=markov_json_path)
        agent = PrefAgent(data['tokens'], app, policy)
        agent = get_agent(
            type=data['agent'], budget=data['tokens'], app=app, policy=policy)
        reporter.add_agent(agent)

        a2d_q = Pipe(i)
        d2a_q = Pipe(i)

        dp.connect(a2d_q, d2a_q, agent.weight)
        agent.connect(d2a_q, a2d_q)
        agents.append(agent)
        t = threading.Thread(target=agent.run)
        t.start()
        threads.append(t)

    t = threading.Thread(target=sched.run)
    t.start()
    threads.append(t)

    for t in threads:
        t: threading.Thread
        t.join()

    reporter.prepare_report()


if __name__ == "__main__":
    data = parse_args(sys.argv[1:])
    main(data)
