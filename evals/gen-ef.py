
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import script_utils
import seaborn as sns

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
ITERATIONS = 2999


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    q_length = data[:, 0]
    assignment = data[:, 1]
    arrival = data[:, 3]
    dep = data[:, 4]
    return (assignment, arrival, dep, q_length)


def get_utility(target_agent: int,
                assignments: np.ndarray, arrivals: np.ndarray, deps: np.ndarray,
                qs: np.ndarray):
    us = list()
    for i in range(ITERATIONS):
        a = (assignments[i] == target_agent)
        u = np.minimum(qs[i - 1] + arrivals[i], a * deps[i])
        u = u[u > 0]
        u = u.sum()
        us.append(u)
    return np.array(us).mean()


def main(agent_num, c_num, util, weights, sched, dd=None, app_type_id=0):
    path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/{sched}_scheduler/queue_q{app_type_id + 1}/'
    if dd is not None:
        path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}/{sched}_scheduler/queue_q{app_type_id + 1}/'
    assignments = np.zeros((ITERATIONS, c_num))
    arrivals = [None for _ in range(agent_num)]
    deps = [None for _ in range(agent_num)]
    qs = [None for _ in range(agent_num)]
    classes = script_utils.get_classes_from_weight_text(weights)
    if len(classes) > 1:
        ws = script_utils.get_agents_weights(agent_num, indices, classes)
        ws = ws / ws.sum()
        ws = ws.reshape(1, agent_num)
    else:
        ws = np.ones((1, agent_num))
        ws = ws / ws.sum()

    for _, dirs, _ in os.walk(path):
        for dir in dirs:
            for subdir, _, files in os.walk(path + dir + '/'):
                agent_id = int(dir.split('agent_')[-1])
                app_assignments = np.zeros((ITERATIONS, c_num))
                app_arrivals = np.zeros((ITERATIONS, c_num))
                app_deps = np.zeros((ITERATIONS, c_num))
                app_q = np.zeros((ITERATIONS, c_num))
                for file in files:
                    if 'app_' in file:
                        app_id = int(file[:-4].split('_')[1])
                        assignment, arrival, dep, q = read_data(subdir, file)
                        app_assignments[:, app_id] = assignment
                        app_arrivals[:, app_id] = arrival
                        app_deps[:, app_id] = dep
                        app_q[:, app_id] = q
                assignments += app_assignments * agent_id
                arrivals[agent_id] = app_arrivals.copy()
                deps[agent_id] = app_deps.copy()
                qs[agent_id] = app_q.copy()
    data = np.zeros((agent_num, agent_num))
    for main_agent_id in range(agent_num):
        for target_agent_id in range(agent_num):
            u = get_utility(target_agent_id, assignments,
                               arrivals[main_agent_id], deps[main_agent_id], qs[main_agent_id])
            u = u * ws[0][main_agent_id] / ws[0][target_agent_id]
            data[main_agent_id][target_agent_id] = u
        # data[main_agent_id] -= data[main_agent_id].min()
        # data[main_agent_id] /= (data[main_agent_id][main_agent_id] - data[main_agent_id].min())
        data[main_agent_id] /= (data[main_agent_id][main_agent_id])
        data[main_agent_id] *= 100
        print('{', end='')
        for target_agent_id in range(agent_num):
            if target_agent_id == agent_num - 1:
                print(f'{data[main_agent_id][target_agent_id]:.2f}', end='')
                break
            print(f'{data[main_agent_id][target_agent_id]:.2f}', end=',')
        print('},',)
        # print('===================')
    plt.figure()
    plt.imshow(data, cmap='YlGnBu', interpolation='nearest')
    plt.colorbar()
    plt.savefig(os.path.join(path, 'ef.png'))
    np.savetxt(os.path.join(path, 'ef.csv'), 
               data, fmt='%.2f', delimiter=',')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = list(script_utils.SCHED_TITLES.keys())
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='1')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-s', '--sched', type=str, default='m_dice')
    parser.add_argument('-d', '--deadline', type=int, default=None)
    parser.add_argument('-a', '--app_type_id', type=int, default=0)
    
    args = parser.parse_args()
    agent_num, indices, c_num, util, weight_text, sched, dd, app_type_id= args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.sched, args.deadline, args.app_type_id
    main(agent_num, c_num, util, weight_text, sched, dd=dd, app_type_id=app_type_id)