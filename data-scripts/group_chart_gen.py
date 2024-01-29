
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
utils = ['40', '50', '60', '70', '80']
utils_title = ['40%', '50%', '60%', '70%', '80%']
scheds = ['g_fair', 'themis', 'mtf']
deadlines = [2, 5, 10, 15, 20, 30]


SCHED_TITLES = {
    'g_fair': 'Gandiva_fair',    
    'themis': 'Themis',    
    'mtf': 'MTF',    
}


def main_dd(agent_num, c_num, util, weight_text):
    means = {}
    stds = {}
    bar_width = 0.2
    plt.figure(figsize=(10, 4))
    xlabels1 = [i for i in range(len(deadlines))]
    xlabels2 = [i + bar_width for i in xlabels1]
    xlabels3 = [i + bar_width for i in xlabels2]
    xlables = [xlabels1, xlabels2, xlabels3]
    for dd in deadlines:
        main_path = root + f'/logs/w_{weight_text}-dd/{agent_num}-{c_num}-{util}util-{weight_text}-dd{dd}/'

        data_file = os.path.join(main_path, 'welfare_data.csv')
        data = np.genfromtxt(data_file, delimiter=",")
        for row_index, _ in enumerate(data):
            sched = scheds[row_index]
            d = [util, scheds[row_index], data[row_index][0], data[row_index][1]]
            if sched not in means.keys():
                means[sched] = []
                stds[sched] = []
            means[sched].append(data[row_index][0])
            stds[sched].append(data[row_index][1] / math.sqrt(50) * 0.95)
    for idx, sched in enumerate(scheds):
        plt.bar(xlables[idx], height=means[sched], width=bar_width, yerr=stds[sched], label=SCHED_TITLES[sched], capsize=2)
    plt.xticks([r + bar_width for r in range(len(xlabels2))], deadlines)
    plt.ylabel('Weighted Social Welfare')
    plt.legend()
    main_path = root + f'/logs/w_{weight_text}-dd'
    plt.savefig(os.path.join(main_path, 'sw_bars.svg'))
    plt.savefig(os.path.join(main_path, 'sw_bars.png'))
    plt.savefig(os.path.join(main_path, 'sw_bars.pdf'))

        


def main(agent_num, c_num, weight_text):
    means = {}
    stds = {}
    bar_width = 0.2
    plt.figure(figsize=(10, 4))
    xlabels1 = [i for i in range(len(utils))]
    xlabels2 = [i + bar_width for i in xlabels1]
    xlabels3 = [i + bar_width for i in xlabels2]
    xlables = [xlabels1, xlabels2, xlabels3]
    for util in utils:
        main_path = root + f'/logs/w_{weight_text}/{agent_num}-{c_num}-{util}util-{weight_text}/'

        data_file = os.path.join(main_path, 'welfare_data.csv')
        data = np.genfromtxt(data_file, delimiter=",")
        for row_index, _ in enumerate(data):
            sched = scheds[row_index]
            d = [util, scheds[row_index], data[row_index][0], data[row_index][1]]
            if sched not in means.keys():
                means[sched] = []
                stds[sched] = []
            means[sched].append(data[row_index][0])
            stds[sched].append(data[row_index][1] / math.sqrt(50) * 0.95)
    for idx, sched in enumerate(scheds):
        plt.bar(xlables[idx], height=means[sched], width=bar_width, yerr=stds[sched], label=SCHED_TITLES[sched], capsize=2)
    plt.xticks([r + bar_width for r in range(len(xlabels2))], utils_title)
    plt.ylabel('Weighted Social Welfare')
    plt.legend()
    main_path = root + f'/logs/w_{weight_text}'
    plt.savefig(os.path.join(main_path, 'sw_bars.svg'))
    plt.savefig(os.path.join(main_path, 'sw_bars.png'))
    plt.savefig(os.path.join(main_path, 'sw_bars.pdf'))

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-c', '--num_clusters', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=bool, default=False)
    args = parser.parse_args()
    agent_num, c_num, weight_text, util, deadline = args.agent_num, args.num_clusters, args.weights, args.util, args.deadline
    if deadline:
        main_dd(agent_num, c_num, util, weight_text)
    else:
        main(agent_num, c_num, weight_text)


