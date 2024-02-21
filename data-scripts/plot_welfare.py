
import argparse
import json
import os, sys
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)

from sys_setup import get_agent_split_indices
from script_utils import get_agents_weights
import script_utils


root = os.path.dirname(os.path.abspath(__file__)) + '/..'
colors = script_utils.COLORS
legends = []


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    return data[:, 0]


def scheduler_welfare_plot_d(agent_num, sched, weights, color, c_num, util, weight_text, dd=None):
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}/{sched}_scheduler/queue_q1/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/{sched}_scheduler/queue_q1/'
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T * weights
    data = data.sum(axis=1)
    mean = []
    std = []
    for index in range(1, len(data) - 1):
        mean.append(data[max(0, index - 50): index].mean())
        std.append(data[max(0, index - 50): index].std())
    mean = np.array(mean)
    std = np.array(std)
    plt.fill_between(range(len(mean)), mean - std, mean + std, color=color, alpha=0.2)
    plt.plot(mean, color=color)
    global legends
    pop_a = patches.Patch(color=color, label=sched)
    legends.append(pop_a)
    
def scheduler_welfare_plot_b(agent_num, sched, weights, c_num, util, weight_text, dd=None):
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}/{sched}_scheduler/queue_q1/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/{sched}_scheduler/queue_q1/'
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    # data = np.array(data).T * 1. / agent_num
    data = np.array(data).T * weights
    data = data.mean(axis=0)
    return data.sum(), data.std()

def d_save_files(main_path):
    plt.legend(handles=legends)
    plt.savefig(os.path.join(main_path, 'd_welfare.svg'))
    plt.savefig(os.path.join(main_path, 'd_welfare.pdf'))
    plt.savefig(os.path.join(main_path, 'd_welfare.png'))
    plt.close()
    

def b_save_files(main_path):
    plt.legend(handles=legends)
    plt.savefig(os.path.join(main_path, 'b_welfare.svg'))
    plt.savefig(os.path.join(main_path, 'b_welfare.pdf'))
    plt.savefig(os.path.join(main_path, 'b_welfare.png'))
    plt.close()
    
def save_data(main_path, means, stds):
    data = np.array([means, stds]).T
    np.savetxt(os.path.join(main_path, 'welfare_data.csv'),
               data, fmt='%.2f', delimiter=',')

def plot_welfare(agent_num, scheds, weights, c_num, util, weight_text, dd=None):
    plt.figure()
    plt.ylabel('Weighted Social Welfare')
    for idx, sched in enumerate(scheds):
        scheduler_welfare_plot_d(agent_num, sched, weights, colors[idx], c_num, util, weight_text, dd)

    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/'
    d_save_files(main_path)
    
    plt.figure()
    plt.ylabel('Unweighted Social Welfare')
    means = []
    stds = []
    for idx, sched in enumerate(scheds):
        mean, std = scheduler_welfare_plot_b(agent_num, sched, weights, c_num, util, weight_text, dd)
        means.append(mean)
        stds.append(std)
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/'
    plt.bar(scheds, means, color=colors)
    b_save_files(main_path)
    save_data(main_path, means, stds)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'wrr', 'ceei', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=str, default=None)
    args = parser.parse_args()
    num_agents, indices, c_num, queue_util, weight_text, dd= args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline
    
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    classes = config['weight_of_classes']
    
    if indices is None:
        indices = get_agent_split_indices(num_agents, classes)
    ws = get_agents_weights(num_agents, indices, classes)
    ws = ws / ws.sum()
    ws = ws.reshape(1, num_agents)
    plot_welfare(num_agents, scheds, ws, c_num, queue_util, weight_text, dd)

