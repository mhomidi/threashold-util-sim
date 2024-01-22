
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    return data[-1000:].mean()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    args = parser.parse_args()
    for sched in ['mtf', 'g_fair', 'themis']:
        path1 = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
            num=args.agent_num, sched=sched)
        data_sh = []
        for subdir, dirs, files in os.walk(path1):
            for file in files:
                # print(os.path.join(subdir, file))
                if file == 'avg_utility.csv':
                    data_sh.append(read_data(subdir, file))
        data_sh = np.array(data_sh)
        print(len(data_sh))

        path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
            num=args.agent_num, sched='rr')
        data_ex = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                if file == 'avg_util.csv':
                    data_ex.append(read_data(subdir, file))
        data_ex = np.array(data_ex)
        data = data_sh / data_ex
        data.sort()
        y = (np.arange(20) + 1) / 20.
        plt.plot(data, y)

    plt.legend(['MTF', 'Gandiva_fair', 'RR'])
    plt.xlabel('$\phi$')
    plt.ylabel('Fraction of Agents')
    plt.savefig(os.path.join(root + '/logs/{num}_agents/'.format(num=args.agent_num), 'si.svg'))
    plt.savefig(os.path.join(root + '/logs/{num}_agents/'.format(num=args.agent_num), 'si.pdf'))
    plt.close()
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
