
import argparse
import json
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


def main(agent_num, sched, c_num, util, weights_text, dd=None, app_sub_id=0):
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}/{sched}_scheduler/queue_q{app_sub_id + 1}/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}-dd{dd}/{sched}_scheduler/queue_q{app_sub_id + 1}/'
    data = []
    for subdir, dirs, files in os.walk(main_path):
        for file in files:
            # print(os.path.join(subdir, file))
            if file == 'utility.csv':
                data.append(read_data(subdir, file))
    data = np.array(data).T
    print(data.shape)
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    data_min = data.min(axis=1)
    data_max = data.max(axis=1)
    plt.plot(mean)
    # plt.fill_between(range(len(data)), mean - std, mean + std, alpha=0.3)
    plt.fill_between(range(len(data)), data_min, data_max, alpha=0.3)
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
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=str, default=None)
    parser.add_argument('-s', '--sched', type=str, default='mtf')
    args = parser.parse_args()
    sched = args.sched
    agent_num, indices, c_num, queue_util, weight_text, dd= args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline
    
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    classes = config['weight_of_classes']


    main(agent_num, sched, c_num, queue_util, weight_text, dd)