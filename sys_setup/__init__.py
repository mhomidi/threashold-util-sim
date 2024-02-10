import random
from modules.applications.dd_queue import DeadlineQueueApplication
from modules.applications.queue import QueueApplication
from modules.applications.dist_app import DistQueueApp
from modules.utils.load_utils import *
from utils.distribution import PoissonGenerator
from modules.scheduler import mtf_scheduler, g_fair_scheduler, rr_scheduler, themis_scheduler 
from modules.coordination import Coordinator, Worker
from modules.policies import ac_policy, ftf_policy, g_fair_policy, fixed_thr_policy
from modules.agents import ac_agent, ftf_agent, g_fair_agent
import time
import json
import numpy as np
import argparse
import os
import torch
import sys
from multiprocessing import Process, Queue

project_src_path = os.path.dirname(os.path.abspath(__file__)) + "/../"


def get_speed_up_factors(speed_ups, sp_weights, num_clusters, num_agents):
    ws = sp_weights / sp_weights.sum()
    sp_factors = np.random.choice(speed_ups, size=(num_agents, num_clusters), p=ws)
    return sp_factors


def create_dist_app(app_type, app_sub_type, config, load_calculator, num_clusters,
                    agent_id, speed_up_factors, arr_rate_coeff, queue_app_type, deadline):
    if app_type == "queue":
        arrival_tps = config["queue_app_arrival_tps"][app_sub_type] * arr_rate_coeff
        departure_tps = config["queue_app_departure_tps"][app_sub_type]
        max_queue_length = config["queue_app_max_queue_length"][app_sub_type]
        alpha = config["queue_app_alpha"]
        apps = list()
        lb_type = config['lb_type']
        if lb_type == 'p2c':
            load_balancer = PowerOfTwoChoices(load_calculator)
        elif lb_type == 'random':
            load_balancer = RandomLoadBalancer(load_calculator)
        else:
            raise Exception('Wrong load balancer type')

        arrival_gen = PoissonGenerator(arrival_tps)
        for j in range(num_clusters):
            depart_gen = PoissonGenerator(departure_tps * speed_up_factors[j])
            if queue_app_type == "without_deadline":
                app = QueueApplication(max_queue_length, depart_gen, alpha)
            elif queue_app_type == "with_deadline":
                app = DeadlineQueueApplication(max_queue_length, depart_gen, alpha, deadline)
            else:
                raise Exception('Wrong queue app type type')
            apps.append(app)

        dist_app = DistQueueApp(agent_id, apps, arrival_gen, load_balancer, queue_app_type)
        return dist_app
    else:
        sys.exit("Unknown app type: {}".format(app_type))


def get_agent_split_indices(num_agents, agents_per_class: dict):
    sum_agent_per_class = float(np.array(agents_per_class).sum())
    index_agent_per_class = 0
    indices = []
    for val in agents_per_class:
        index_agent_per_class += val
        idx = index_agent_per_class / sum_agent_per_class
        if len(indices) == 0 or idx != indices[-1]:
            indices.append(int(index_agent_per_class / sum_agent_per_class * num_agents))
    indices.pop()
    return indices


def get_agents_weights(num_agents, agent_split_indices, weight_of_classes):
    agent_weights = np.ones(num_agents)
    assert len(agent_split_indices) + 1 == len(weight_of_classes)
    ids_weight_classes = np.split(
        np.arange(0, num_agents), agent_split_indices)
    for wc, wpc in zip(ids_weight_classes, weight_of_classes):
        agent_weights[wc] *= wpc
    return agent_weights


def get_arrival_rate_coefficients(num_agents, agent_split_indices, arrival_rate_coefficient_of_classes):
    agent_arr_coeff = np.ones(num_agents)
    assert len(agent_split_indices) + 1 == len(arrival_rate_coefficient_of_classes)
    ids_agent_coeffs = np.split(
        np.arange(0, num_agents), agent_split_indices)
    for ac, acc in zip(ids_agent_coeffs, arrival_rate_coefficient_of_classes):
        agent_arr_coeff[ac] *= acc
    return agent_arr_coeff


def main(config_file_name, app_type_id, app_sub_type_id, policy_id, scheduler_id, threshold_in=-1, queue_app_type='wo_dd', themis_type_id=1):
    start_time = time.time()
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    folder_name = project_src_path
    num_workers = config["num_workers"]
    num_agents = config["num_agents"]
    app_type = config["app_types"][app_type_id]
    queue_app_type = config["queue_app_type"][queue_app_type]
    deadline = config["deadline"]
    assert app_sub_type_id < len(config["app_sub_types"][app_type])
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    policy_type = config["policy_types"][policy_id]
    scheduler_type = config["scheduler_types"][scheduler_id]
    coordinator_step_print = config["coordinator_step_print"]
    token_coefficient = config["token_coefficient"]
    num_clusters = config["num_clusters"]
    num_iterations = config["num_iterations"]
    ac_policy_config = config['ac_policy_config']
    a_h1_size = ac_policy_config['a_h1_size']
    c_h1_size = ac_policy_config['c_h1_size']
    c_h2_size = ac_policy_config['c_h2_size']
    threshold_steps = ac_policy_config['threshold_steps']
    actor_net_type = ac_policy_config['actor_net_type']
    mini_batch_size = ac_policy_config['mini_batch_size']
    ac_discount_factor = config['ac_discount_factor'][app_type][app_sub_type]
    ac_a_lr = config['a_lr'][app_type][app_sub_type]
    ac_c_lr = config['c_lr'][app_type][app_sub_type]
    std_max = config['std_max'][app_type][app_sub_type]
    speed_ups = np.array(config['speed_ups'])
    sp_weights = np.array(config['sp_weights'])
    agents_per_class = config["agents_per_class"]
    weight_of_classes = config['weight_of_classes']
    arrival_rate_coefficient_of_classes = config['arrival_rate_coefficient_of_classes']
    themis_type = config['themis_type'][themis_type_id]

    agent_split_indices = get_agent_split_indices(num_agents, agents_per_class)

    agent_weights = get_agents_weights(num_agents, agent_split_indices, weight_of_classes)
    agent_weights /= agent_weights.sum()

    sp_factors = get_speed_up_factors(speed_ups, sp_weights, num_clusters, num_agents)

    arr_rate_coeffs = get_arrival_rate_coefficients(
        num_agents, agent_split_indices, arrival_rate_coefficient_of_classes)
    

    util = config["queue_util"]
    classes = config['weight_of_classes']
    weights_text = "".join([str(item) for item in classes])
    path = f"{folder_name}/logs/{num_agents}-{num_clusters}-{util}util-{weights_text}/{scheduler_type}/{app_type}_{app_sub_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]

    agents_list = []
    worker_processors: list[Process] = []

    if scheduler_type == "mtf_scheduler":
        scheduler = mtf_scheduler.MTFScheduler(
            agent_weights, num_agents, num_clusters, token_coefficient)
    elif scheduler_type == "g_fair_scheduler":
        scheduler = g_fair_scheduler.GFairScheduler(
            agent_weights, num_agents, num_clusters)
    elif scheduler_type == "rr_scheduler":
        scheduler = rr_scheduler.RoundRobinScheduler(
            agent_weights, num_agents, num_clusters)
    elif scheduler_type == "themis_scheduler":
        departure_rates = sp_factors * config['queue_app_departure_tps'][app_sub_type]
        arrival_rates = np.ones((num_agents, 1)) * config['queue_app_arrival_tps'][app_sub_type]
        arrival_rates = arrival_rates * arr_rate_coeffs.reshape((num_agents, 1))
        if themis_type == 'queue_length':
            scheduler = themis_scheduler.QLengthFairScheduler(
                agent_weights, num_agents, num_clusters, departure_rates, arrival_rates)
        elif themis_type == 'throughput':
            scheduler = themis_scheduler.ThroughputFairScheduler(
                agent_weights, num_agents, num_clusters, departure_rates, arrival_rates)
    else:
        sys.exit("unknown scheduler type")

    coordinator = Coordinator(scheduler, num_iterations, num_agents,
                              num_clusters, num_workers, w2c_queues, c2w_queues, coordinator_step_print)
    coordinator_processor = Process(target=coordinator.run, args=())
    coordinator_processor.start()

    for agent_id in range(num_agents):
        agent_spf = sp_factors[agent_id]
        if policy_type == "ac_policy":
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(
                app_type, app_sub_type, config, load_calculator, num_clusters,
                agent_id, agent_spf, arr_rate_coeffs[agent_id], queue_app_type, deadline)
            policy = ac_policy.ACPolicy(a_input_size=(num_agents + num_clusters), c_input_size=(num_agents + num_clusters),
                                        a_h1_size=a_h1_size, c_h1_size=c_h1_size, c_h2_size=c_h2_size,
                                        a_lr=ac_a_lr, c_lr=ac_c_lr, df=ac_discount_factor, std_max=std_max, num_clusters=num_clusters,
                                        mini_batch_size=mini_batch_size, threshold_steps=threshold_steps,
                                        actor_net_type=actor_net_type)
            agent = ac_agent.ACAgent(
                agent_id, agent_weights[agent_id], dist_app, policy, scheduler.tokens.copy())
        elif policy_type == "thr_policy":
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(
                app_type, app_sub_type, config, load_calculator, num_clusters,
                agent_id, agent_spf, arr_rate_coeffs[agent_id], queue_app_type, deadline)
            threshold = threshold_in
            if threshold_in == -1:
                threshold = config["threshold"][app_type][app_sub_type]
            policy = fixed_thr_policy.FixedThrPolicy(num_clusters, threshold)
            agent = ac_agent.ACAgent(
                agent_id, agent_weights[agent_id], dist_app, policy, scheduler.tokens.copy())
        elif policy_type == "g_fair_policy":
            load_calculator = GFairLoadCalculator()
            dist_app = create_dist_app(
                app_type, app_sub_type, config, load_calculator, num_clusters,
                agent_id, agent_spf, arr_rate_coeffs[agent_id], queue_app_type, deadline)
            policy = g_fair_policy.GFairPolicy(num_clusters)
            agent = g_fair_agent.GFairAgent(
                agent_id, agent_weights[agent_id], dist_app, policy)
        elif policy_type == "themis_policy":
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(
                app_type, app_sub_type, config, load_calculator, num_clusters,
                agent_id, agent_spf, arr_rate_coeffs[agent_id], queue_app_type, deadline)
            policy = ftf_policy.ThemisPolicy(num_clusters)
            agent = ftf_agent.ThemisAgent(
                agent_id, agent_weights[agent_id], dist_app, policy)
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
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)

    parser = argparse.ArgumentParser()
    parser.add_argument('app_type_id', type=int)
    parser.add_argument('app_type_sub_id', type=int)
    parser.add_argument('policy_id', type=int)
    parser.add_argument('scheduler_id', type=int)
    args = parser.parse_args()
    main(config_file_path, args.app_type_id,
         args.app_type_sub_id, args.policy_id, args.scheduler_id)
    print("Done")

# python test/test.py 1 0 0 0 ## queue, q0, AC
# python test/test.py 1 0 1 1 ## queue, q0, G-fair
# python test/test.py 1 0 1 2 ## queue, q0, RR
# python test/test.py 1 0 2 3 ## queue, q0, Themis
