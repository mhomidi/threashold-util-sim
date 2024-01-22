
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'

seen_agents = None


def read_data(direct, file, indices):
    agent_id = str.split(direct, 'agent_')[-1]
    try:
        agent_id = int(agent_id)
    except:
        return
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    group = -1
    count = 0
    for idx in indices:
        if agent_id < idx:
            group = count
            break
        count += 1
    return data[:, 0], group


def main(agent_num, sched, indices):
    legs = ['w=1', 'w=2', 'w=3']
    main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
        num=agent_num, sched=sched)
    data = [list() for _ in range(len(indices) + 1)]
    global seen_agents
    seen_agents = [False for _ in range(agent_num)]
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            # print(os.path.join(subdir, file))
            if file == 'utility.csv':
                d, g = read_data(subdir, file, indices)
                data[g].append(d)
    plt.figure()
    plt.title('Average Accumulated Queue Length for Agnet Classes')
    for item in data:
        # print(len(item))
        m = np.array(item).T.mean(axis=1)
        plt.plot(m)
    plt.legend(legs)

    plt.savefig(os.path.join(main_path, 'classes_utils.svg'))
    plt.savefig(os.path.join(main_path, 'classes_utils.pdf'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-s', '--sched', type=str, default='mtf')
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=[10, 15])
    args = parser.parse_args()
    agent_num, sched, indices = args.agent_num, args.sched, args.indices
    main(agent_num, sched, indices)
