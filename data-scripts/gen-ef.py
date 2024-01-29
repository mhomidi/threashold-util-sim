
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
ITERATIONS = 999


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    assignment = data[:, 1]
    arrival = data[:, 3]
    dep = data[:, 4]
    return (assignment, arrival, dep)


def get_utility(target_agent: int,
                assignments: np.ndarray, arrivals: np.ndarray, deps: np.ndarray):
    q = np.zeros(arrivals[0].shape)
    us = list()
    for i in range(ITERATIONS):
        a = (assignments[i] == target_agent)
        # print(q + arrivals[i], q, arrivals[i])
        u = np.minimum(q + arrivals[i], a * deps[i])
        u = u[u >= 0]
        u = u.sum()
        q = np.maximum(0, q + arrivals[i] - a * deps[i])
        us.append(u)
    return np.array(us).mean()


def main(agent_num, c_num, util, weights, sched):
    path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/{sched}_scheduler/queue_q1/'
    data = []
    assignments = np.zeros((ITERATIONS, c_num))
    arrivals = [None for _ in range(agent_num)]
    deps = [None for _ in range(agent_num)]

    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            for subdir, _, files in os.walk(path + dir + '/'):
                agent_id = int(dir.split('agent_')[-1])
                app_assignments = np.zeros((ITERATIONS, c_num))
                app_arrivals = np.zeros((ITERATIONS, c_num))
                app_deps = np.zeros((ITERATIONS, c_num))
                for file in files:
                    if 'app_' in file:
                        app_id = int(file[:-4].split('_')[1])
                        # print(app_id)
                        assignment, arrival, dep = read_data(subdir, file)
                        app_assignments[:, app_id] = assignment
                        app_arrivals[:, app_id] = arrival
                        app_deps[:, app_id] = dep
                assignments += app_assignments * agent_id
                arrivals[agent_id] = app_arrivals.copy()
                deps[agent_id] = app_deps.copy()
    for main_agent_id in range(agent_num):
        for target_agent_id in range(agent_num):
            u = get_utility(target_agent_id, assignments,
                               arrivals[main_agent_id], deps[main_agent_id])
            # print(f'a{main_agent_id} --> a{target_agent_id}: {u:.2f}')
            print(f'{u:.2f}')
        print('===================')


# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
        
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_clusters', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='1')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-s', '--sched', type=str, default='mtf')
    args = parser.parse_args()
    agent_num, indices, c_num, util, weight_text, sched= args.agent_num, args.indices, args.num_clusters, args.util, args.weights, args.sched
    main(agent_num, c_num, util, weight_text, sched)