import random
from utils.constructors import *
import config
import threading
from utils.report import *
from utils.pipe import Pipe, Pipe
from modules.coordination import Coordinator, Worker
from modules.applications.dist_app import DistributedApplication
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
        'lambda': 3,
        'thr': 0.0,
    }

    opts, _ = getopt.getopt(argv, "h:s:p:a:l:r:g", [
                            "scheduler=", "policy=", "app=", "lambda=", "thr=", "agent="])
    for opt, arg in opts:
        if opt == '-h':
            print(
                'test.py -s <scheduler> -p <policy> -a <app_type> -l <lambda> -r <thr> -g <agent>')
            sys.exit()
        elif opt in ("-s", "--sched"):
            data['sched'] = arg
        elif opt in ('-p', '--policy'):
            data['policy'] = arg
        elif opt in ('-a', '--app'):
            data['app'] = arg
        elif opt in ('-l', '--lambda'):
            data['lambda'] = arg
        elif opt in ('-r', '--thr'):
            data['thr'] = arg
        elif opt in ('-g', '--agent'):
            data['agent'] = arg

    return data


def main(data: dict):
    sched = get_scheduler(type=data['sched'])

    agents = list()
    workers = list()
    worker_threads = list()

    coordinator = Coordinator(sched)
    last_agent_index = 0
    default_weight = 1. / config.get(config.AGENT_NUM)

    for i in range(config.get(config.WORKER_NUM)):
        worker = Worker(i)
        workers.append(worker)

        for i in range(config.get(config.WORKER_AGENT_NUM)):
            if last_agent_index > config.get(config.AGENT_NUM):
                break
            policy = get_policy(type=data['policy'], budget=data['tokens'])
            cluster_apps = list()
            for i in range(config.get(config.CLUSTER_NUM)):
                app = get_application(
                    type=data['app'], markov_json_path=markov_json_path)
                cluster_apps.append(app)
            app = DistributedApplication(applications=cluster_apps)
            agent = Agent(agent_id=last_agent_index, weight=default_weight,
                          distributed_app=app, policy=policy)
            agents.append(agent)
            last_agent_index += 1

        coordinator.add_pipes(worker.w2c_pipe, worker.c2w_pipe)
        t = threading.Thread(target=worker.run)
        t.start()
        worker_threads.append(t)

    t = threading.Thread(target=coordinator.run)
    t.start()
    worker_threads.append(t)

    for t in worker_threads:
        t: threading.Thread
        t.join()


if __name__ == "__main__":
    data = parse_args(sys.argv[1:])
    main(data)
