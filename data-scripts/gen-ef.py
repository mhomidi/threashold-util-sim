
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    assignment = data[:, 1]
    arrival = data[:, -2]
    dep = data[:, -1]
    return (assignment, arrival, dep)


def get_utility(target_agent: int,
                assignments: np.ndarray, arrivals: np.ndarray, deps: np.ndarray):
    q = np.zeros(10)
    us = list()
    for i in range(999):
        a = (assignments[i] == target_agent)
        # print(a * deps[i], a, deps[i])
        # print(q + arrivals[i], q, arrivals[i])
        u = q + arrivals[i] - a * deps[i]
        u = u[u >= 0]
        u = -u.sum()
        q = np.maximum(0, q + arrivals[i] - a * deps[i])
        # print(u, a, q)
        us.append(u)
    return np.array(us[-200:]).mean()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    parser.add_argument('sched', type=str)
    args = parser.parse_args()
    agent_num = args.agent_num
    root += '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=args.agent_num, sched=args.sched)
    data = []
    assignments = np.zeros((999, 10))
    arrivals = [None for _ in range(agent_num)]
    deps = [None for _ in range(agent_num)]

    for _, dirs, _ in os.walk(root):
        for dir in dirs:
            for subdir, _, files in os.walk(root + dir + '/'):
                agent_id = int(dir[-1])
                app_assignments = np.zeros((999, 10))
                app_arrivals = np.zeros((999, 10))
                app_deps = np.zeros((999, 10))
                for file in files:
                    if 'app_' in file:
                        app_id = int(file[-5])
                        assignment, arrival, dep = read_data(subdir, file)
                        app_assignments[:, app_id] = assignment
                        app_arrivals[:, app_id] = arrival
                        app_deps[:, app_id] = dep
                assignments += app_assignments * agent_id
                arrivals[agent_id] = app_arrivals.copy()
                deps[agent_id] = app_deps.copy()

    for main_agent_id in range(agent_num):
        for target_agent_id in range(agent_num):
            util = get_utility(target_agent_id, assignments,
                               arrivals[main_agent_id], deps[main_agent_id])
            print('a{ma:d} --> a{tar}: {u:d}'.format(ma=main_agent_id,
                  tar=target_agent_id, u=int(util)))
        print('===================')

# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
