
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
colors = ['deepskyblue', 'orange', 'darkolivegreen', 'violet']


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    return data[:, 0]


def scheduler_welfare_plot(agent_num, sched, weights, color):
    main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=agent_num, sched=sched)
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T * weights
    mean = data.sum(axis=1)
    plt.plot(mean, color=color)
    

def save_files(main_path, scheds):
    plt.legend(scheds)
    plt.savefig(os.path.join(main_path, 'welfare.svg'))
    plt.savefig(os.path.join(main_path, 'welfare.pdf'))
    plt.savefig(os.path.join(main_path, 'welfare.png'))
    plt.close()
    
def plot_welfare(agent_num, scheds, weights):
    plt.figure()
    plt.title('Social Welfare')
    for idx, sched in enumerate(scheds):
        scheduler_welfare_plot(agent_num, sched, weights, colors[idx])
    main_path = root + f'/logs/{agent_num}_agents'
    save_files(main_path, scheds)