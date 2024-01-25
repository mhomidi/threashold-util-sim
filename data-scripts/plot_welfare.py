
import json
import os, sys
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)

from sys_setup import get_agent_split_indices, get_agents_weights


root = os.path.dirname(os.path.abspath(__file__)) + '/..'
colors = ['deepskyblue', 'orange', 'darkolivegreen', 'violet']
legends = []


def read_data(direct, file):
    # res = []
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    # for idx, datum in enumerate(data):
    #     res.append(datum[min(0, idx - 50), idx])
    return data[:, 0]


def scheduler_welfare_plot_d(agent_num, sched, weights, color):
    main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=agent_num, sched=sched)
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
    


def scheduler_welfare_plot_b(agent_num, sched, weights):
    main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=agent_num, sched=sched)
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T * weights
    data = data.sum(axis=1)
    return data[-50:].mean()
    
    

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
    
def plot_welfare(agent_num, scheds, weights):
    plt.figure()
    plt.title('Social Welfare')
    plt.ylabel('Weighted Average')
    for idx, sched in enumerate(scheds):
        scheduler_welfare_plot_d(agent_num, sched, weights, colors[idx])
    main_path = root + f'/logs/{agent_num}_agents'
    d_save_files(main_path)
    
    plt.figure()
    plt.title('Social Welfare')
    plt.ylabel('Weighted Average')
    bar_vals = []
    for idx, sched in enumerate(scheds):
        bar_vals.append(scheduler_welfare_plot_b(agent_num, sched, weights))
    main_path = root + f'/logs/{agent_num}_agents'
    plt.bar(scheds, bar_vals)
    b_save_files(main_path)
    

if __name__ == "__main__":
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        
    num_agents = config["num_agents"]
    agents_per_class = config['agents_per_class']
    indices = get_agent_split_indices(num_agents, agents_per_class)
    classes = config['weight_of_classes']
    scheds = ['g_fair', 'themis', 'mtf']
    
    ws = get_agents_weights(num_agents, indices, classes)
    ws = ws / ws.sum()
    ws = ws.reshape(1, num_agents)
    plot_welfare(num_agents, scheds, ws)