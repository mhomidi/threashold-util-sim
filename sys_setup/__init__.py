import random
from modules.applications.dd_queue import DeadlineQueueApplication
from modules.applications.queue import QueueApplication
from modules.applications.dist_app import DistQueueApp
from modules.utils.load_utils import *
from utils.distribution import PoissonGenerator
from modules.scheduler import l_dice_scheduler, m_dice_scheduler, s_dice_scheduler, ss_scheduler, tant_scheduler
from modules.coordination import Coordinator, Worker
from modules.policies import Policy
from modules.agents import Agent
import time
import json
import numpy as np
import argparse
import os
import torch
import sys
from multiprocessing import Process, Queue

project_src_path = os.path.dirname(os.path.abspath(__file__)) + "/../"


def get_speed_up_factors(speed_ups, sp_weights, num_nodes, num_agents):
    ws = sp_weights / sp_weights.sum()
    sp_factors = np.zeros((num_agents, num_nodes))
    start_index = 0
    for idx, w in enumerate(ws):
        end_index = int(w * num_nodes + start_index)
        if idx == len(ws) - 1:
            sp_factors[:, start_index:] = speed_ups[idx]
        else:
            sp_factors[:, start_index:end_index] = speed_ups[idx]
        start_index = end_index
    return sp_factors


def create_dist_app(app_type, app_sub_type, config, load_calculator, num_nodes,
                    agent_id, speed_up_factors, arr_rate_coeff, queue_app_type, deadline,
                    max_queue_coeff):
    if app_type == "queue":
        arrival_tps = config["queue_app_arrival_tps"][app_sub_type] * arr_rate_coeff
        departure_tps = config["queue_app_departure_tps"]
        max_queue_length = int(config["queue_app_max_queue_length"] * max_queue_coeff)
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
        for j in range(num_nodes):
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

def get_coefficients(config, app_sub_type):
    num_agents = config["num_agents"]
    agents_per_class = config["agents_per_class"]
    agent_split_indices = get_agent_split_indices(num_agents, agents_per_class)
    
    agent_arr_coeff = np.ones(num_agents)
    agent_weights = np.ones(num_agents)
    agent_mqs = np.ones(num_agents)
    weight_of_classes = config['weight_of_classes']
    for w in weight_of_classes:
        assert(isinstance(w, int))
    max_queue_length_coefficient_of_classes = config['max_queue_length_coefficient_of_classes']
    for mqc in max_queue_length_coefficient_of_classes:
        assert(isinstance(mqc, float) or isinstance(w, int))
    assert len(agent_split_indices) + 1 == len(weight_of_classes)
    assert len(agent_split_indices) + 1 == len(max_queue_length_coefficient_of_classes)

    ids_agent_coeffs = np.split(
        np.arange(0, num_agents), agent_split_indices)
    for ac, wc, mqc in zip(ids_agent_coeffs, weight_of_classes, max_queue_length_coefficient_of_classes):
        agent_arr_coeff[ac] *= wc
        agent_weights[ac] *= wc
        agent_mqs[ac] *= mqc
        
    return agent_arr_coeff, agent_weights, agent_mqs


def main(config_file_name, app_type_id, app_sub_type_id, policy_id, scheduler_id, threshold_in=-1, deadline_type='wo_dd', themis_type_id=1):
    start_time = time.time()
    with open(config_file_name, 'r') as f:
        config = json.load(f)
    folder_name = project_src_path
    num_workers = config["num_workers"]
    num_agents = config["num_agents"]
    policy_types = config['policy_types']
    app_type = config["app_types"][app_type_id]
    deadline_type = config["queue_app_type"][deadline_type]
    deadline = config["deadline"]
    assert app_sub_type_id < len(config["app_sub_types"][app_type])
    app_sub_type = config["app_sub_types"][app_type][app_sub_type_id]
    policy_type = config["policy_types"][policy_id]
    scheduler_type = config["scheduler_types"][scheduler_id]
    coordinator_step_print = config["coordinator_step_print"]
    num_nodes = config["num_nodes"]
    num_iterations = config["num_iterations"]
    speed_ups = np.array(config['speed_ups'])
    sp_weights = np.array(config['sp_weights'])
    weight_of_classes = config['weight_of_classes']
    for w in weight_of_classes:
        assert(isinstance(w, int))
    themis_type = config['themis_type'][themis_type_id]
    
    arr_rate_coeffs, agent_weights, max_queue_coeffs = get_coefficients(config, app_sub_type)
    non_normalized_agent_weights = agent_weights.copy()
    agent_weights /= agent_weights.sum()

    sp_factors = get_speed_up_factors(speed_ups, sp_weights, num_nodes, num_agents)
    

    util = config["queue_util"]
    classes = config['weight_of_classes']
    weights_text = "".join([str(item) for item in classes])
    path = f"{folder_name}/logs/{num_agents}-{num_nodes}-{util}util-{weights_text}/{scheduler_type}/{app_type}_{app_sub_type}"
    if deadline_type == 'with_deadline':
        path = f"{folder_name}/logs/{num_agents}-{num_nodes}-{util}util-{weights_text}-dd{deadline}/{scheduler_type}/{app_type}_{app_sub_type}"
    if not os.path.exists(path):
        os.makedirs(path)

    w2c_queues = [Queue() for _ in range(num_workers)]
    c2w_queues = [Queue() for _ in range(num_workers)]

    agents_list = []
    worker_processors: list[Process] = []

    if scheduler_type == "g_fair_scheduler":
        scheduler = ss_scheduler.GFairScheduler(
            agent_weights, num_agents, num_nodes)
    elif scheduler_type == "l_dice_scheduler":
        scheduler = l_dice_scheduler.LotteryScheduler(
            non_normalized_agent_weights, num_agents, num_nodes)
    elif scheduler_type == "s_dice_scheduler":
        scheduler = s_dice_scheduler.StrideScheduler(
            non_normalized_agent_weights, num_agents, num_nodes)
    elif scheduler_type == "themis_scheduler":
        departure_rates = sp_factors * config['queue_app_departure_tps']
        arrival_rates = np.ones((num_agents, 1)) * config['queue_app_arrival_tps'][app_sub_type]
        arrival_rates = arrival_rates * arr_rate_coeffs.reshape((num_agents, 1))
        if themis_type == 'queue_length':
            scheduler = tant_scheduler.QLengthFairScheduler(
                agent_weights, num_agents, num_nodes, departure_rates, arrival_rates)
        elif themis_type == 'throughput':
            scheduler = tant_scheduler.ThroughputFairScheduler(
                agent_weights, num_agents, num_nodes, departure_rates, arrival_rates)
        else:
            sys.exit("Unknown themis scheduler type!")
    elif scheduler_type == "m_dice_scheduler":
        scheduler = m_dice_scheduler.CEEIScheduler(
            agent_weights, num_agents, num_nodes)
    else:
        sys.exit("unknown scheduler type!")

    coordinator = Coordinator(scheduler, num_iterations, num_agents,
                              num_nodes, num_workers, w2c_queues, c2w_queues, coordinator_step_print)
    coordinator_processor = Process(target=coordinator.run, args=())
    coordinator_processor.start()

    for agent_id in range(num_agents):
        agent_spf = sp_factors[agent_id]
        if policy_type in policy_types:
            load_calculator = ExpectedWaitTimeLoadCalculator()
            dist_app = create_dist_app(
                app_type, app_sub_type, config, load_calculator, num_nodes,
                agent_id, agent_spf, arr_rate_coeffs[agent_id], deadline_type, deadline, 
                max_queue_coeffs[agent_id])
            policy = Policy(num_nodes)
            agent = Agent(agent_id, agent_weights[agent_id], dist_app, policy)
        else:
            sys.exit("Unknown policy type!")

        agents_list.append(agent)

    ids_list = np.array_split(np.arange(0, num_agents), num_workers)
    for worker_id in range(0, num_workers):
        worker = Worker(agents_list[ids_list[worker_id][0]:ids_list[worker_id]
                                    [-1] + 1], num_nodes, w2c_queues[worker_id], c2w_queues[worker_id])
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
