
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

def main(agent_num, sched):
    main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=agent_num, sched=sched)
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            # print(os.path.join(subdir, file))
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    data_min = data.min(axis=1)
    data_max = data.max(axis=1)
    plt.plot(mean)
    plt.fill_between(range(len(data)), mean - std, mean + std, alpha=0.3)
    # plt.fill_between(range(len(data)), data_min, data_max, alpha=0.3)
    plt.savefig(os.path.join(main_path, 'avg_util.svg'))
    plt.savefig(os.path.join(main_path, 'avg_util.pdf'))
    np.savetxt(os.path.join(main_path, 'avg_util.csv'),
               data.mean(axis=1), fmt='%d', delimiter=',')

    np.savetxt(os.path.join(main_path, 'std_util.csv'),
               data.std(axis=1), fmt='%d', delimiter=',')
    plt.close()

# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    parser.add_argument('sched', type=str)
    args = parser.parse_args()
    main(args.agent_num, args.sched)