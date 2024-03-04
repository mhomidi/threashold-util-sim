
import argparse
import json
import os
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import script_utils 

root = os.path.dirname(os.path.abspath(__file__)) + '/..'

colors = script_utils.COLORS
sched_titles = script_utils.SCHED_TITLES

def main(agent_num, title, is_std, c_num, util, weights_text, dd=None, scheds = script_utils.SCHED_TITLES.keys(), app_sub_id=0):
    means = []
    stds = []
    plt.figure()
    legs = []
    for sched in scheds:
        subdir_sys = f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}'
        if dd is not None:
            subdir_sys = f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}-dd{dd}'
        subdir_sched = f'/{sched}_scheduler/queue_q{app_sub_id + 1}/'
        direct = root + subdir_sys + subdir_sched
        file = os.path.join(direct, 'avg_util.csv')
        mean = pd.read_csv(file).values[:, 0]
        means.append(mean)
        file = os.path.join(direct, 'std_util.csv')
        std = pd.read_csv(file).values[:, 0]
        stds.append(std)

    if is_std:
        for idx, sched in enumerate(scheds):
            plt.fill_between(range(len(std)), means[idx]-stds[idx], means[idx]+stds[idx], alpha=0.2, color=colors[idx])
            legs.append(patches.Patch(color=colors[idx], label=sched_titles[sched]))

    plt.legend(handles=legs)

    for idx, sched in enumerate(scheds):
        plt.plot(means[idx], label=sched, color=colors[idx])
    plt.xlabel('Iterations')
    plt.ylabel(title)
    plt.savefig(os.path.join(root + subdir_sys, 'all_plot.svg'))
    plt.savefig(os.path.join(root + subdir_sys, 'all_plot.pdf'))
    plt.savefig(os.path.join(root + subdir_sys, 'all_plot.png'))
    plt.close()

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    scheds = list(script_utils.SCHED_TITLES.keys())
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-t', '--title', type=str, default='Average Throughput')
    parser.add_argument('-v', '--is_std', type=bool, default=True)
    parser.add_argument('-d', '--deadline', type=str, default=None)
    args = parser.parse_args()
    title = args.title
    is_std = args.is_std
    agent_num, indices, c_num, queue_util, weight_text, dd= args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline
    
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    classes = config['weight_of_classes']
    scheds = list(script_utils.SCHED_TITLES.keys())

    main(agent_num, title, is_std, c_num, queue_util, weight_text, dd, scheds)