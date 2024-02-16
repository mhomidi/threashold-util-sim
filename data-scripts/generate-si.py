
import argparse
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)

from sys_setup import get_agents_weights

scheds = ['g_fair', 'themis', 'mtf']
scheds_titles = ['Gandiva_fair', 'Themis', 'MTF']

def get_weight_class(weights):
    ws = []
    while weights > 0:
        ws.append(weights % 10)
        weights /= 10
        weights = int(weights)
    return ws[::-1]


def read_data(direct, file, indices):
    agent_id = str.split(direct, 'agent_')[-1]
    try:
        agent_id = int(agent_id)
    except:
        return
    data_file = os.path.join(direct, file)
    data = pd.read_csv(data_file).values
    if indices is None:
        return data[-50:, 0].mean(), 0
    group = -1
    count = 0
    for idx in indices:
        if agent_id < idx:
            group = count
            break
        count += 1
    return data[-50:, 0].mean(), group

def main(agent_num, indices, c_num, util, weights):
    ws = get_agents_weights(agent_num, indices, get_weight_class(int(weights)))
    norm_ws = ws / ws.sum()
    ex_t = 48*norm_ws*c_num
    for sched in scheds:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/{sched}_scheduler/queue_q1/'
        if dd is not None:
            main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}-dd{dd}/{sched}_scheduler/queue_q1/'
        if indices is None:
            data = [[]]
        else:
            data = [list() for _ in range(len(indices) + 1)]
        global seen_agents
        seen_agents = [False for _ in range(agent_num)]
        for subdir, dirs, files in os.walk(main_path):
            for file in files:
                if file == 'utility.csv':
                    d, g = read_data(subdir, file, indices)
                    data[g].append(d)
        draft = []
        for d in data:
            draft += d
        sh_t = np.array(draft)
        phi = sh_t / ex_t
        phi.sort()
        y = (np.arange(20) + 1) / 20.
        plt.plot(phi, y)

    plt.legend(scheds_titles)
    plt.xlabel('$\phi$')
    plt.ylabel('Fraction of Agents')
    plt.savefig(os.path.join(root + f'/logs/{agent_num}-{c_num}-{util}util-{weights}/', 'si.svg'))
    plt.close()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_nodes', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-d', '--deadline', type=str, default=None)
    args = parser.parse_args()
    agent_num, indices, c_num, util, weight_text, dd= args.agent_num, args.indices, args.num_nodes, args.util, args.weights, args.deadline
    main(agent_num, indices, c_num, util, weight_text)