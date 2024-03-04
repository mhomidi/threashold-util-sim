
import argparse
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches 
import pandas as pd
import script_utils

root = os.path.dirname(os.path.abspath(__file__)) + '/..'

seen_agents = None
scheds = script_utils.SCHED_TITLES.keys()
scheds_title = script_utils.SCHED_TITLES

def create_weight_labels(weights):
    ws = []
    while weights > 0:
        ws.append(weights % 10)
        weights /= 10
        weights = int(weights)
    ws_text = ['W=' + str(w) for w in ws[::-1]]
    return ws_text, ws[::-1]

def read_data(direct, file, indices):
    agent_id = str.split(direct, 'agent_')[-1]
    try:
        agent_id = int(agent_id)
    except:
        return
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    if indices is None:
        return data[:, 0], 0
    group = -1
    count = 0
    for idx in indices:
        if agent_id < idx:
            group = count
            break
        count += 1
    return data[:, 0], group

def plot_average_plot_per_w(agent_num, indices, c_num, util, weights, dd=None, app_type_id=0):
    ws_text, ws = create_weight_labels(int(weights))
    means = {}
    cil = {}
    for w in ws_text:
        means[w] = []
        cil[w] = []
    for sched in scheds:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/{sched}_scheduler/queue_q{app_type_id + 1}/'
        if dd is not None:
            main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}/{sched}_scheduler/queue_q{app_type_id+1}/'
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
        for w_idx, item in enumerate(data):
            if len(data) > 1:
                d = np.array(item).T.mean(axis=1)
            else:
                d = np.array(item).T.mean(axis=1)
            m = d.mean()
            s = d.std()
            means[ws_text[w_idx]].append(m)
            cil[ws_text[w_idx]].append(s / math.sqrt(50) * 0.95)
    
    plt.figure()
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}/'
    bar_width = 0.2
    xs = [[i + j*bar_width for i in range(len(scheds))] for j in range(len(ws_text))]
    for idx, w in enumerate(ws_text):
        plt.bar(xs[idx], height=means[w], width=bar_width, yerr=cil[w], label=w, capsize=2)
    nw = len(ws_text)
    plt.xticks([r + ((nw-1) * bar_width / 2) for r in range(len(scheds_title))], scheds_title.values())
    plt.ylabel('Average Throughput')
    plt.legend()
    plt.savefig(os.path.join(main_path, 'weight_bar.svg'))
    plt.savefig(os.path.join(main_path, 'weight_bar.pdf'))
    plt.savefig(os.path.join(main_path, 'weight_bar.png'))
    plt.close()
    data = []
    for idx, w in enumerate(ws_text):
        data.append(means[w])
    data = np.array(data)
    for index, _ in enumerate(data):
        data[index] /= data[0][0]
    data = data.T
    np.savetxt(os.path.join(main_path, 'weight_throughput.csv'),
               data, fmt='%.2f', delimiter='  & ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = script_utils.SCHED_TITLES.keys()
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=str, default=None)
    parser.add_argument('-a', '--app_type_id', type=int, default=0)
    args = parser.parse_args()
    agent_num, indices, c_num, util, weight_text, dd, app_type_id = args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline, args.app_type_id
    plot_average_plot_per_w(agent_num, indices, c_num, util, weight_text, dd, app_type_id)