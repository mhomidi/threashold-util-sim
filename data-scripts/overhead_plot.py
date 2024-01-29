import os
import matplotlib.pyplot as plt
import numpy as np
import math


data_astat = {
    '20A-20C': [0.1885, 0.2725, 368.9075],
    '20A-40C': [0.2365, 0.358, 755.7875],
    '20A-60C': [0.2693, 0.6863, 1208.9039]
}

data_cstat = {
    '20A-40C': [0.2365, 0.358, 755.7875],
    '40A-40C': [0.3161, 0.4934, 1636.3396],
    '60A-40C': [0.3603, 0.5069, 2624.8415],
}

scheds = ['Gandiva_fair', 'Themis', 'MTF']
log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../logs'

def correct_data(data:dict):
    for key in data.keys():
        data[key] = data[key][1:] + data[key][:1]

if __name__ == '__main__':
    correct_data(data_astat)
    correct_data(data_cstat)

    bar_width = 0.2
    xs = [[i + j*bar_width for i in range(3)] for j in range(3)]
    for idx, key in enumerate(data_astat.keys()):
        plt.bar(xs[idx], height=np.array(data_astat[key]), width=bar_width, label=key, capsize=2)
    nw = len(data_astat.keys())
    plt.xticks([r + ((nw-1) * bar_width / 2) for r in range(len(scheds))], scheds)
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Avg. Scheduling Time (ms)')
    plt.savefig(os.path.join(log_dir, 'overhead_astat.svg'))
    plt.savefig(os.path.join(log_dir, 'overhead_astat.png'))
    plt.close()



    bar_width = 0.2
    xs = [[i + j*bar_width for i in range(3)] for j in range(3)]
    for idx, key in enumerate(data_cstat.keys()):
        plt.bar(xs[idx], height=np.array(data_cstat[key]), width=bar_width, label=key, capsize=2)
    nw = len(data_cstat.keys())
    plt.xticks([r + ((nw-1) * bar_width / 2) for r in range(len(scheds))], scheds)
    plt.legend()
    plt.yscale('log')
    plt.ylabel('Avg. Scheduling Time (ms)')
    plt.savefig(os.path.join(log_dir, 'overhead_cstat.svg'))
    plt.savefig(os.path.join(log_dir, 'overhead_cstat.png'))
    plt.close()
