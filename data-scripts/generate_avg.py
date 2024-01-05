
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    avgs = list()
    for i in range(1, len(data)):
        avgs.append(data[max(0, i - 100): i].mean())
    data = np.array(avgs)
    plt.figure()
    plt.plot(data)
    plt.savefig(os.path.join(direct, 'avg_utility.png'))
    np.savetxt(os.path.join(direct, 'avg_utility.csv'),
               data, fmt='%d', delimiter=',')
    plt.close()
    return avgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    parser.add_argument('sched', type=str)
    args = parser.parse_args()
    root += '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=args.agent_num, sched=args.sched)
    data = []
    for subdir, dirs, files in os.walk(root):
        for file in files:
            # print(os.path.join(subdir, file))
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T
    plt.plot(data.mean(axis=1))
    plt.savefig(os.path.join(root, 'avg_util.png'))
    np.savetxt(os.path.join(root, 'avg_util.csv'),
               data.mean(axis=1), fmt='%d', delimiter=',')
    plt.close()

# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
