
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
    args = parser.parse_args()
    system = args.sys
    agent_num = args.agent_num

    # scheds = ['mtf', 'g_fair', 'rr', 'themis']
    scheds = ['mtf', 'g_fair', 'rr']
    avgs = []
    plt.figure()
    subdir_sys = '/logs/{num}_{sys}_agents'
    if system is None:
        subdir_sys = '/logs/{num}_agents'

    subdir_sched = '/{sched}_scheduler/queue_q1/'
    for sched in scheds:
        direct = (root + subdir_sys + subdir_sched).format(
            num=agent_num, sched=sched, sys=system)
        file = os.path.join(direct, 'avg_util.csv')
        data = pd.read_csv(file).values
        avgs.append(data)
        plt.plot(data, label=sched)

    plt.legend(scheds)

    plt.xlabel('Iterations')
    plt.ylabel('Avg. Utility')
    plt.savefig(os.path.join((root + subdir_sys).format(num=agent_num, sys=system), 'all_plot.svg'))

    plt.close()
