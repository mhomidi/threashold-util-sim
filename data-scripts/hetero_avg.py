
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches 

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
    if indices is None:
        return data[:, 0], 0
    group = -1
    count = 0
    for idx in indices:
        if agent_id < idx:
            group = count
            break
        count += 1
    return data[:, 0], group

def plot_per_class(agent_num, indices, a_class_index, a_class, scheds, title, c_num, queue_util, weights_text, dd=None):
    colors = ['purple', 'orange', 'green']
    res = dict()
    for sched in scheds:
        
        main_path = root + f'/logs/{agent_num}-{c_num}-{queue_util}util-{weights_text}/{sched}_scheduler/queue_q1/'
        if dd is not None:
            main_path = root + f'/logs/{agent_num}-{c_num}-{queue_util}util-{weights_text}-dd{dd}/{sched}_scheduler/queue_q1/'
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
        res[sched] = data[a_class_index]
    plt.figure()
    plt.ylabel(title)
    legs = []
    for idx, key in enumerate(res.keys()):
        d = np.array(res[key]).T
        m = d.mean(axis=1)
        low_bound = d.mean(axis=1) - d.std(axis=1)
        high_bound = d.mean(axis=1)  + d.std(axis=1)
        plt.plot(m, color=colors[idx])
        plt.fill_between(range(len(d)), low_bound, high_bound, color=colors[idx], alpha=0.3)
        pop_a = mpatches.Patch(color=colors[idx], label=scheds[idx])
        legs.append(pop_a)
    plt.legend(handles=legs)
    main_path = root + f'/logs/{agent_num}-{c_num}-{queue_util}util-{weights_text}/'
    filename = 'classes_utils_{c}'.format(c=a_class)
    plt.savefig(os.path.join(main_path, filename + '.svg'))
    plt.savefig(os.path.join(main_path, filename + '.pdf'))
    plt.savefig(os.path.join(main_path, filename + '.png'))
    plt.close()
        

def plot_per_sched(agent_num, sched, indices, c_num, util, weights_text, dd=None):
    legs = ['w=1', 'w=2', 'w=3']
    main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}/{sched}_scheduler/queue_q1/'
    if dd is not None:
        main_path = root + f'/logs/{agent_num}-{c_num}-{util}util-{weights_text}-dd{dd}/{sched}_scheduler/queue_q1/'
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
    plt.figure()
    plt.ylabel('Average Throughput')
    for item in data:
        m = np.array(item).T.mean(axis=1)
        plt.plot(m)
    plt.legend(legs)

    plt.savefig(os.path.join(main_path, 'classes_all_utils.svg'))
    plt.savefig(os.path.join(main_path, 'classes_all_utils.pdf'))
    plt.savefig(os.path.join(main_path, 'classes_all_utils.png'))
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    parser.add_argument('-c', '--num_clusters', type=int, default=40)
    parser.add_argument('-w', '--weights', type=str, default='124')
    parser.add_argument('-u', '--util', type=str, default='80')
    parser.add_argument('-v', '--is_std', type=bool, default=True)
    parser.add_argument('-d', '--deadline', type=str, default=None)
    parser.add_argument('-s', '--sched', type=str, default='mtf')
    parser.add_argument('-f', '--func', type=str, default='per_sched')
    parser.add_argument('-ac', '--agent_class', type=int, default=1)
    parser.add_argument('-ci', '--class_index', type=int, default=0)
    parser.add_argument('-tt', '--title', type=str, default='No Title')
    args = parser.parse_args()
    title = args.title
    func = args.func
    agent_class = args.agent_class
    sched = args.sched
    class_index = args.class_index
    agent_num, indices, c_num, queue_util, weight_text, dd= args.agent_num, args.indices, args.num_clusters, args.util, args.weights, args.deadline
    
    if func == 'per_sched':
        plot_per_sched(agent_num, sched, indices, c_num, queue_util, weight_text, dd)
    if func == 'per_class':
        plot_per_class(agent_num, indices, class_index, agent_class, scheds, title, c_num, queue_util, weight_text, dd)