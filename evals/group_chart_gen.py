
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import script_utils

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
utils = list(script_utils.UTILS.keys())
utils_title = list(script_utils.UTILS.values())
scheds = list(script_utils.SCHED_TITLES.keys())


sched_titles = script_utils.SCHED_TITLES


def main_dd(agent_num, c_num, util, weight_text):
    means = {}
    stds = {}
    bar_width = 0.12
    plt.figure(figsize=(10, 4))
    xlables = []
    for idx in range(len(scheds)):
        ls = [i + idx * bar_width for i in range(len(script_utils.DEADLINES))]
        xlables.append(ls)
        
    for dd in script_utils.DEADLINES:
        main_path = root + f'/logs/w_{weight_text}-dd/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/'

        data_file = os.path.join(main_path, 'welfare_data.csv')
        data = np.genfromtxt(data_file, delimiter="  & ").T
        for row_index, _ in enumerate(data):
            sched = scheds[row_index]
            # d = [util, scheds[row_index], data[row_index][0], data[row_index][1]]
            if sched not in means.keys():
                means[sched] = []
                stds[sched] = []
            means[sched].append(data[row_index][0] / data[0][0])
            stds[sched].append(data[row_index][1]/ (data[0][1] * math.sqrt(50)) * 0.95)
    for idx, sched in enumerate(scheds):
        plt.bar(xlables[idx], height=means[sched], width=bar_width, yerr=stds[sched], label=sched_titles[sched], capsize=2)
    mid = len(scheds) * bar_width / 2.0
    plt.xticks([r + mid for r in range(len(xlables[0]))], script_utils.DEADLINES)
    plt.ylabel('Weighted Social Welfare')
    plt.legend()
    main_path = root + f'/logs/w_{weight_text}-dd'
    plt.savefig(os.path.join(main_path, 'sw_bars.svg'))
    plt.savefig(os.path.join(main_path, 'sw_bars.png'))
    plt.savefig(os.path.join(main_path, 'sw_bars.pdf'))
    # print(list(means.values()))
    # print(list(stds.values()))
    data = np.array(list(means.values())).T
    np.savetxt(os.path.join(main_path, 'throughput.csv'),
               data, fmt='%.2f', delimiter='  & ')
        


def main(agent_num, c_num, weight_text):
    means = {}
    stds = {}
    bar_width = 0.2
    plt.figure(figsize=(10, 4))
    xlables = []
    for idx in range(len(scheds)):
        ls = [i + idx * bar_width for i in range(len(script_utils.DEADLINES))]
        xlables.append(ls)
    for util in utils:
        main_path = root + f'/logs/w_{weight_text}/{agent_num}-{c_num}-{util}util-{weight_text}-dd5/'

        data_file = os.path.join(main_path, 'welfare_data.csv')
        data = np.genfromtxt(data_file, delimiter="  & ").T
        for row_index, _ in enumerate(data):
            sched = scheds[row_index]
            if sched not in means.keys():
                means[sched] = []
                stds[sched] = []
            means[sched].append(data[row_index][0] / data[0][0])
            stds[sched].append(data[row_index][1] / (data[0][1] * math.sqrt(50)) * 0.95)
    print(data.shape, len(means))
    for idx, sched in enumerate(scheds):
        plt.bar(xlables[idx], height=means[sched], width=bar_width, yerr=stds[sched], label=sched_titles[sched], capsize=2)
    plt.xticks([r + bar_width for r in range(len(xlables[0]))], utils_title)
    plt.ylabel('Weighted Social Welfare')
    plt.legend()
    main_path = root + f'/logs/w_{weight_text}'
    plt.savefig(os.path.join(main_path, 'sw_bars.svg'))
    plt.savefig(os.path.join(main_path, 'sw_bars.png'))
    plt.savefig(os.path.join(main_path, 'sw_bars.pdf'))
    data = np.array(list(means.values())).T
    np.savetxt(os.path.join(main_path, 'throughput.csv'),
               data, fmt='%.2f', delimiter='  & ')

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=bool, default=False)
    args = parser.parse_args()
    agent_num, c_num, weight_text, util, deadline = args.agent_num, args.num_nodes, args.weights, args.util, args.deadline
    if deadline:
        main_dd(agent_num, c_num, util, weight_text)
    else:
        main(agent_num, c_num, weight_text)


