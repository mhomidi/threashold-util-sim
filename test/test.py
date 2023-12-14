from modules.applications.queue import QueueApplication
from modules.applications.dist_app import DistQueueApp
from modules.utils.load_utils import *
from utils.distribution import UniformGenerator, PoissonGenerator
from modules.scheduler import mtf_scheduler, g_fair_scheduler
from modules.coordination import Coordinator, Worker
from modules.policies import ac_policy, g_fair_policy, fixed_thr_policy
from modules.agents import ac_agent, g_fair_agent
import time
import json
import numpy as np
import argparse
import os
import sys
from multiprocessing import Process, Queue

project_src_path = os.path.dirname(os.path.abspath(__file__)) + "/../"


def create_dist_app(app_type, app_sub_type, config, load_calculator, num_clusters, agent_id):
    if app_type == "queue":
        arrival_tps = config["queue_app_arrival_tps"][app_sub_type]
        departure_tps = config["queue_app_departure_tps"][app_sub_type]
        max_queue_length = config["queue_app_max_queue_length"][app_sub_type]
        avg_throughput_alpha = config["queue_app_avg_throughput_alpha"]
        apps = list()
        load_balancer = RandomLoadBalancer()
        arrival_gen = PoissonGenerator(arrival_tps)
        for j in range(num_clusters):
            depart_gen = PoissonGenerator(departure_tps)
            app = QueueApplication(max_queue_length, depart_gen, avg_throughput_alpha, load_calculator)
            apps.append(app)

        dist_app = DistQueueApp(agent_id, apps, arrival_gen, load_balancer)
        return dist_app
    else:
        sys.exit("Unknown app type: {}".format(app_type))

def main(config_file_name, app_type_id, app_sub_type_id, policy_id, threshold_in=-1, weights=-1):
    start_time = time.time()
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    folder_name = project_src_path
    num_workers = config["num_workers"]
    num_agents = config["num_agents"]
    app_type = config["app_types"][app_type_id]
    assert app_sub_type_id < len(config["app_sub_types"][app_type])
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    policy_type = config["policy_types"][policy_id]
    scheduler_type = config["scheduler_types"][policy_id]
    app_utilities = config["app_utilities"]
    token_coefficient = config["token_coefficient"]
    num_clusters = config["num_clusters"]
    num_iterations = config["num_iterations"]

    agent_weights = np.ones(num_agents)
    # TODO test the implementation above
    # TODO make sure weights are not assumed to be fractional anywhere in the project

    if weights == -1:
        num_weight_classes = config["num_weight_classes"]
        weight_per_class = config["weight_per_class"]
        assert num_weight_classes == len(weight_per_class)
        ids_weight_classes = np.split(np.arange(0, num_agents), num_weight_classes)
        for wc, wpc in zip(ids_weight_classes, weight_per_class):
            agent_weights[wc] *= wpc

    path = f"{folder_name}/logs/{num_agents}_agents/{policy_type}/{app_type}_{app_sub_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]

    agents_list = []
    worker_processors: list[Process] = []

    if scheduler_type == "mtf_scheduler":
        scheduler = mtf_scheduler.MTFScheduler(agent_weights, num_agents, num_clusters, token_coefficient)
    elif scheduler_type == "g_fair_scheduler":
        scheduler = g_fair_scheduler.GFairScheduler(agent_weights, num_agents, num_clusters)
    else:
        sys.exit("unknown scheduler type")

    coordinator = Coordinator(scheduler, num_iterations, num_agents, num_clusters, num_workers, w2c_queues, c2w_queues)
    coordinator_processor = Process(target=coordinator.run, args=())
    coordinator_processor.start()

    for agent_id in range(num_agents):
        if policy_type == "ac_policy":
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(app_type, app_sub_type, config, load_calculator, num_clusters, agent_id)
            # TODO read parameters from config and set them accordingly
            policy = ac_policy.ACPolicy(num_clusters)
            agent = ac_agent.ACAgent(agent_id, agent_weights[agent_id], dist_app, policy, scheduler.tokens.copy())
        elif policy_type == "thr_policy":
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(app_type, app_sub_type, config, load_calculator, num_clusters, agent_id)
            threshold = threshold_in
            if threshold_in == -1:
                threshold = config["threshold"][app_type][app_sub_type]
            policy = fixed_thr_policy.FixedThrPolicy(num_clusters, threshold)
            agent = ac_agent.ACAgent(agent_id, agent_weights[agent_id], dist_app, policy, scheduler.tokens.copy())
        elif policy_type == "g_fair_policy":
            load_calculator = GFairLoadCalculator()
            dist_app = create_dist_app(app_type, app_sub_type, config, load_calculator, num_clusters, agent_id)
            policy = g_fair_policy.GFairPolicy(num_clusters)
            agent = g_fair_agent.GFairAgent(agent_id, agent_weights[agent_id], dist_app, policy)
        else:
            sys.exit("Unknown policy type!")

        agents_list.append(agent)

    ids_list = np.array_split(np.arange(0, num_agents), num_workers)
    for worker_id in range(0, num_workers):
        worker = Worker(agents_list[ids_list[worker_id][0]:ids_list[worker_id]
                                                           [-1] + 1], num_clusters, w2c_queues[worker_id], c2w_queues[worker_id])
        worker_processor = Process(target=worker.run, args=(path,))
        worker_processors.append(worker_processor)
        worker_processor.start()

    for worker_processor in worker_processors:
        worker_processor.join()

    coordinator_processor.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time} seconds")


if __name__ == "__main__":
    config_file_path = project_src_path + "config/sys_config_default.json"

    parser = argparse.ArgumentParser()
    parser.add_argument('app_type_id', type=int)
    parser.add_argument('app_type_sub_id', type=int)
    parser.add_argument('policy_id', type=int)
    args = parser.parse_args()
    main(config_file_path, args.app_type_id,
         args.app_type_sub_id, args.policy_id, -1)
    print("Done")

# python test/test.py 1 0 0 --> queue, q0, AC
# python test/test.py 1 0 1 --> queue, q0, G-fair
