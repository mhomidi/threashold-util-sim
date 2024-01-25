
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches 
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

def plot_average_plot_per_w(agent_num, indices):
    ws = ['w=1', 'w=2', 'w=4']
    scheds = ['g_fair', 'themis', 'mtf']
    raw_df = []
    
    for sched in scheds:
        main_path = root + '/logs/{num}_agents/{sched}_scheduler/queue_q1/'.format(
            num=agent_num, sched=sched)
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
        for w_idx, item in enumerate(data):
            m = np.array(item).T.mean(axis=1)
            m = np.array(m[-50:]).mean()
            raw_df.append([sched, ws[w_idx], m])
    plt.figure()
    main_path = root + f'/logs/{agent_num}_agents/'
    df = pd.DataFrame(raw_df, columns=['Scheduler', 'Weight', 'Mean'])
    df.pivot(index='Scheduler', columns='Weight', values='Mean').plot(kind='bar')
    plt.ylabel('Average Throughput')
    plt.savefig(os.path.join(main_path, 'weight_bar.svg'))
    plt.savefig(os.path.join(main_path, 'weight_bar.pdf'))
    plt.savefig(os.path.join(main_path, 'weight_bar.png'))
    plt.close()
    
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scheds = ['g_fair', 'themis', 'mtf']
    parser.add_argument('-n', '--agent_num', type=int, default=20)
    # parser.add_argument('-s', '--sched', type=str, default='mtf')
    parser.add_argument('-i', '--indices', type=int, nargs='*', default=None)
    # parser.add_argument('-f', '--func', type=str, default='per_sched')
    # parser.add_argument('-ac', '--agent_class', type=int, default=1)
    # parser.add_argument('-tt', '--title', type=str, default='No Title')
    args = parser.parse_args()
    agent_num, indices,= args.agent_num, args.indices
    plot_average_plot_per_w(agent_num, indices)