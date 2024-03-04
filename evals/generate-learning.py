
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    parser.add_argument('sys', type=str)
    args = parser.parse_args()
    agent_num = args.agent_num
    avgs = []
    # plt.figure()
    labels = []
    all_data = []
    for agnet_id in range(agent_num):
        direct = root + '/logs/{num}_{sys}_agents/{sched}_scheduler/queue_q1/agent_{id}'.format(
            num=agent_num, sched='mtf', sys=args.sys, id=agnet_id)
        file = os.path.join(direct, 'avg_utility.csv')
        data = pd.read_csv(file).values
        avgs.append(data)
        # plt.plot(data)
        all_data.append(data)
    all_data = np.array(all_data).T.reshape((3998, 20))[0:3000]
    plt.figure()
    lb = all_data.mean(axis=1) - all_data.std(axis=1)
    ub = all_data.mean(axis=1) + all_data.std(axis=1)
    plt.plot(all_data.mean(axis=1))
    plt.fill_between(range(len(all_data)), lb, ub, alpha=0.3)

    plt.xlabel('Iterations')
    plt.ylabel('Avg. Utility')
    plt.savefig(os.path.join(root + '/results', 'learning-avg.svg'))

    plt.close()
