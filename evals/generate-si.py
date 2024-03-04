
import argparse
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)
ITERATIONS = 2999

from script_utils import get_agents_weights
import script_utils

scheds = list(script_utils.SCHED_TITLES.keys())
scheds_titles = list(script_utils.SCHED_TITLES.values())

def get_weight_class(weights):
    ws = []
    while weights > 0:
        ws.append(weights % 10)
        weights /= 10
        weights = int(weights)
    return ws[::-1]


def read_data(direct, file, indices):
    agent_id = str.split(direct, 'agent_')[-1]
    try:
        agent_id = int(agent_id)
    except:
        return
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    if indices is None:
        return data[:, 0].mean(), 0
    group = -1
    count = 0
    for idx in indices:
        if agent_id < idx:
            group = count
            break
        count += 1
    return data[:, 0].mean(), group



def read_data_for_ex_t(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    # q_length = data[:, 0]
    # assignment = data[:, 1]
    arrival = data[:, 3]
    dep = data[:, 4]
    return (arrival, dep)

def get_utility(arrivals: np.ndarray, deps: np.ndarray):
    us = list()
    q = np.zeros((1, c_num))
    for i in range(ITERATIONS):
        u = np.minimum(arrivals[i], deps[i])
        u = u[u > 0]
        u = u.sum()
        us.append(u)
    return np.array(us).mean()

def generate_ex_t(agent_num, c_num, util, weights, sched, dd=None, app_type_id=0):
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
    else:
        ws = np.ones(agent_num)
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
                        arrival, dep= read_data_for_ex_t(subdir, file)
                        app_arrivals[:, app_id] = arrival
                        app_deps[:, app_id] = dep
                assignments += app_assignments * agent_id
                arrivals[agent_id] = app_arrivals.copy()
                deps[agent_id] = app_deps.copy()
                qs[agent_id] = app_q.copy()

    data = [get_utility(arrivals[agent_id], deps[agent_id]) for agent_id in range(agent_num)]
    return np.array(data) * ws

    



def main(agent_num, indices, c_num, util, weights, app_type_id, dd=None):
    ws = get_agents_weights(agent_num, indices, get_weight_class(int(weights)))
    norm_ws = ws / ws.sum()
    # ex_t = 48*norm_ws*c_num
    for sched in scheds:
        ex_t = generate_ex_t(agent_num, c_num, util, weights, sched, dd, app_type_id)
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/{sched}_scheduler/queue_q{app_type_id + 1}/'
        if dd is not None:
            main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}/{sched}_scheduler/queue_q{app_type_id + 1}/'
        if indices is None:
            data = [[]]
        else:
            data = [list() for _ in range(len(indices) + 1)]
        global seen_agents
        seen_agents = [False for _ in range(agent_num)]
        for subdir, dirs, files in os.walk(main_path):
            for file in files:
                if file == 'utility.csv':
                    d, g = read_data(subdir, file, indices)
                    data[g].append(d)
        draft = []
        for d in data:
            draft += d
        sh_t = np.array(draft) * norm_ws
        phi = sh_t / ex_t
        phi.sort()
        y = (np.arange(20) + 1) / 20.
        plt.plot(phi, y)
        print(sched)
        for i in range(len(phi)):
            print(f'({phi[i]:.2f},{y[i]})', end=' ')
        print('===============')

    plt.legend(scheds_titles)
    plt.xlabel('$\phi$')
    plt.ylabel('Fraction of Agents')
    
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}'
    plt.savefig(os.path.join(main_path, 'si.png'))
    plt.close()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=int, default=None)
    parser.add_argument('-a', '--app_type_id', type=int, default=None)
    args = parser.parse_args()
    agent_num, indices, c_num, util, weight_text, dd, app_type_id = args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline, args.app_type_id
    main(agent_num, indices, c_num, util, weight_text, app_type_id, dd)