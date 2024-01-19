
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


def read_data(direct, file):
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    plt.figure()
    plt.plot(data)
    plt.savefig(os.path.join(direct, 'avg_utility.svg'))
    plt.savefig(os.path.join(direct, 'avg_utility.pdf'))
    plt.close()
    return data[:, 0]


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
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    plt.plot(mean)
    plt.fill_between(range(len(data)), mean - std, mean + std, alpha=0.3)
    plt.savefig(os.path.join(root, 'avg_util.svg'))
    plt.savefig(os.path.join(root, 'avg_util.pdf'))
    np.savetxt(os.path.join(root, 'avg_util.csv'),
               data.mean(axis=1), fmt='%d', delimiter=',')

    np.savetxt(os.path.join(root, 'std_util.csv'),
               data.std(axis=1), fmt='%d', delimiter=',')
    plt.close()

# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
