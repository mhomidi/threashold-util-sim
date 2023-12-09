import argparse
import os
import sys
from multiprocessing import Process, Queue

import numpy as np

import json
import time

from modules.coordination import Coordinator, Worker
from modules.scheduler.mtf_scheduler import MTFScheduler


def main(config_file_name, app_type_id, app_sub_type_id, policy_id, threshold_in):
    start_time = time.time()
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    folder_name = config["dir_name"]
    num_workers = config["num_workers"]
    num_agents = config["num_agents"]
    app_type = config["app_types"][app_type_id]
    assert app_sub_type_id < len(config["app_sub_types"][app_type])
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    policy_type = config["policy_types"][policy_id]
    scheduler_type = config["scheduler_types"][policy_id]
    app_utilities = config["app_utilities"]

    path = f"{folder_name}/{num_agents}_agents/{policy_type}/{app_type}_{app_sub_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]

    agents_list = []
    worker_processors = []

    for i in range(num_agents):
        if app_type == "queue":
            arrival_tps = config["queue_app_arrival_tps"][app_sub_type]
            sprinting_tps = config["queue_app_sprinting_tps"][app_sub_type]
            nominal_tps = config["queue_app_nominal_tps"][app_sub_type]
            max_queue_length = config["queue_app_max_queue_length"][app_sub_type]
            # TODO load balancer and load calculator and generator
            # TODO create apps 
            # TODO create dist app
        # elif app_type == "markov":
        #     transition_matrix = config["markov_app_transition_matrices"][app_sub_type]
        #     app = applications.MarkovApp(transition_matrix, app_utilities, np.random.choice(app_utilities))
        else:
            sys.exit("wrong app type!")

        if policy_type == "ac_policy":
            # a_lr = config["a_lr_no_noise"][app_type][app_sub_type]
            # c_lr = config["c_lr_no_noise"][app_type][app_sub_type]
            # 
            # a_h1_size = config["ac_policy_config"]["a_h1_size"]
            # c_h1_size = config["ac_policy_config"]["c_h1_size"]
            # std_max = config["std_max"][app_type][app_sub_type]
            # df = config["ac_discount_factor"][app_type][app_sub_type]
            # mini_batch_size = config["ac_policy_config"]["mini_batch_size"]
            # TODO: policy = ACPolicy
            # TODO: agent = ACAgent(i, weight, distributed_app, policy, tokens)
        # elif policy_type == "thr_policy":
        #     threshold = threshold_in
        #     if threshold == -1:
        #         threshold = config["threshold"][app_type][app_sub_type]
        #     policy = policies.ThrPolicy(threshold)
        #     server = servers.ThrServer(i, period, policy, app, servers_config, utility_normalization_factor)
        elif policy_type == "g_fair_policy":
            # TODO agent = GFairAgent(i, weight, distributed_app, policy)
        else:
            sys.exit("Wrong policy type!")

        agents_list.append(agent)

    ids_list = np.array_split(np.arange(0, num_agents), num_workers)
    for i in range(0, num_workers):
        worker = Worker(agents_list[ids_list[i][0]:ids_list[i][-1] + 1], w2c_queues[i], c2w_queues[i])
        worker_processor = Process(target=worker.run, args=(path,))
        worker_processors.append(worker_processor)
        worker_processor.start()

    if scheduler_type == "mtf_scheduler":
        token_coefficient = config["token_coefficient"]
        # TODO scheduler = MTFScheduler(agent_weights, num_agents, num_clusters, token_coefficient)
    elif scheduler_type == "g_fair_scheduler":
        # TODO scheduler = GFairScheduler(...)

    # TODO coordinator = Coordinator(scheduler, num_iterations, num_agents, num_workers, w2c_queues, c2w_queues)
    
    coordinator_processor = Process(target=coordinator.run, args=())
    coordinator_processor.start()

    for worker_processor in worker_processors:
        worker_processor.join()

    coordinator_processor.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time} seconds")


if __name__ == "__main__":
    config_file_path = "/Users/smzahedi/Documents/Papers/Pref-GPU-Expr/config/sys_config_default.json"

    parser = argparse.ArgumentParser()
    parser.add_argument('app_type_id', type=int)
    parser.add_argument('app_type_sub_id', type=int)
    parser.add_argument('policy_id', type=int)
    args = parser.parse_args()
    main(config_file_path, args.app_type_id, args.app_type_sub_id, args.policy_id, -1)
    print("Done")
