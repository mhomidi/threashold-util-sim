
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--agent_num', type=int)
    parser.add_argument('-s', '--sys', type=str, default=None)
    parser.add_argument('-t', '--title', type=str, default='Random Load Balancer')
    parser.add_argument('-v', '--is_std', type=bool, default=True)
    args = parser.parse_args()
    system = args.sys
    agent_num = args.agent_num
    title = args.title
    is_std = args.is_std

    # scheds = ['mtf', 'g_fair', 'rr', 'themis']
    # scheds = ['mtf', 'rr', 'themis']
    # scheds = ['mtf', 'g_fair', 'rr']
    scheds = ['g_fair', 'mtf']
    # scheds = ['mtf', 'rr']
    colors = ['deepskyblue', 'orange', 'darkolivegreen', 'violet']
    means = []
    stds = []
    plt.figure()
    plt.title(title)
    subdir_sys = '/logs/{num}_{sys}_agents'
    if system is None:
        subdir_sys = '/logs/{num}_agents'

    subdir_sched = '/{sched}_scheduler/queue_q1/'
    for sched in scheds:
        direct = (root + subdir_sys + subdir_sched).format(
            num=agent_num, sched=sched, sys=system)
        file = os.path.join(direct, 'avg_util.csv')
        mean = pd.read_csv(file).values[:, 0]
        means.append(mean)
        file = os.path.join(direct, 'std_util.csv')
        std = pd.read_csv(file).values[:, 0]
        stds.append(std)
        plt.fill_between(range(len(std)), mean-std, mean+std, alpha=0.3)

    if is_std:
        for idx, sched in enumerate(scheds):
            plt.fill_between(range(len(std)), means[idx]-stds[idx], means[idx]+stds[idx], alpha=0.3, color=colors[idx])

    plt.legend(scheds)

    for idx, sched in enumerate(scheds):
        plt.plot(means[idx], label=sched, color=colors[idx])
    plt.xlabel('Iterations')
    plt.ylabel('Avg. Utility')
    plt.savefig(os.path.join((root + subdir_sys).format(num=agent_num, sys=system), 'all_plot.svg'))
    plt.savefig(os.path.join((root + subdir_sys).format(num=agent_num, sys=system), 'all_plot.pdf'))

    plt.close()
