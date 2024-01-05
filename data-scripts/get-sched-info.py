
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'


# def print_plot(direct, file):
#     path = direct + '/plots'
#     if not os.path.exists(path):
#         os.makedirs(path)
#     data_file = os.path.join(direct, file)
#     data = pd.read_csv(data_file).values
#     avgs = list()
#     for i in range(1, len(data)):
#         avgs.append(data[max(0, i - 500): i].mean())
#     plt.plot(avgs)
#     plt.savefig(os.path.join(path, file) + '_plot.png')
#     plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('agent_num', type=int)
    parser.add_argument('sys', type=str)
    scheds = ['mtf', 'g_fair', 'rr', 'themis']
    args = parser.parse_args()
    for sched in scheds:
        direct = root + '/logs/{num}_{sys}_agents/{sched}_scheduler/queue_q1/'.format(
            num=args.agent_num, sched=sched, sys=args.sys)
        file = os.path.join(direct, 'avg_util.csv')
        data = pd.read_csv(file).values[-500:]
        print('===== {sched} ====='.format(sched=sched))
        print('mean:', int(data.mean()))
        print('std:', data.std())
