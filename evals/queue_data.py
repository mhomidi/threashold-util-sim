
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__)) + '/..'

def print_data(subdir, file, rnd, app_id):
    data_file = os.path.join(subdir, file)
    data = pd.read_csv(data_file).values
    if app_id is None:
        print(file, ':', int(data[rnd, 0]), int(data[rnd, 3]), int(data[rnd, 4]), int(data[rnd, 1]), int(data[rnd, 5]))
    elif '_' + app_id + '.csv' in file:
        # q, arr, dep, assignment, rate
        print(file, ':', int(data[rnd, 0]), int(data[rnd, 3]), int(data[rnd, 4]), int(data[rnd, 1]))
    return data[rnd, 0], data[rnd, 3], data[rnd, 4]


def create_diagram_app_queue(subdir, file, app_id, agent_id):
    if app_id is None:
        raise Exception("Need to specify app")
    data_file = os.path.join(subdir, file)
    data = pd.read_csv(data_file).values
    fig_path = os.path.join(subdir, 'figs')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.figure()
    plt.title("Aggregated Queue Length of App" + app_id + " of A_" + agent_id)
    plt.xlabel('Iter')
    plt.xlabel('Queue')
    plt.plot(data)
    plt.savefig(os.path.join(fig_path, file[:-4] + '.svg'))
    plt.close()


def generate_queues_data(args):
    system = args.sys
    agent_num = args.agent_num
    agent_id = args.agent_id
    sched = args.sched
    rnd = args.round
    app_id = args.app_id

    subdir_sys = '/logs/{num}_{sys}_agents'
    if system is None:
        subdir_sys = '/logs/{num}_agents'

    subdir_sched = '/{sched}_scheduler/queue_q1/agent_{agent_id}'

    direct = (root + subdir_sys + subdir_sched).format(
        num=agent_num, sched=sched, sys=system, agent_id=agent_id)

    data = []
    for subdir, dirs, files in os.walk(direct):
        for file in sorted(files):
            if 'app_' in file:
                data.append(print_data(subdir, file, rnd, app_id))
            if file == 'utility.csv':
                create_diagram_app_queue(subdir, file, app_id, agent_id)

    print(np.array(data).mean(axis=0))


def plot_comparison(args):
    system = args.sys
    agent_num = args.agent_num
    agent_id = args.agent_id
    rnd = args.round
    app_id = args.app_id

    subdir_sys = '/logs/{num}_{sys}_agents'
    if system is None:
        subdir_sys = '/logs/{num}_agents'

    data = []
    for sched in ['rr', 'g_fair']:

        subdir_sched = '/{sched}_scheduler/queue_q1'
        title = "Avg Queue Lengths"
        if app_id is not None:
            subdir_sched = '/{sched}_scheduler/queue_q1/agent_{agent_id}'
            title = "Aggregated Queue Length of App " + app_id + " of Agent " + agent_id

        direct = (root + subdir_sys + subdir_sched).format(
            num=agent_num, sched=sched, sys=system, agent_id=agent_id)

        for subdir, dirs, files in os.walk(direct):
            for file in sorted(files):
                if app_id is not None and file == 'utility.csv':
                    data_file = os.path.join(subdir, file)
                    d = pd.read_csv(data_file).values.T
                    data.append(d[0].tolist())
                elif app_id is None and file == 'avg_util.csv':
                    data_file = os.path.join(subdir, file)
                    d = pd.read_csv(data_file).values.T
                    data.append(d[0].tolist())

    data = np.array(data).T
    print(data.shape)
    plt.figure()
    plt.title(title)
    fig_path = os.path.join(root, 'data-scripts', 'logs', 'figs')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.xlabel('Iter')
    plt.xlabel('Queue')
    plt.plot(data)
    plt.legend(['RR', 'G_fair'])
    plt.savefig(os.path.join(fig_path, 'test.pdf'))
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--agent_num', type=int)
    parser.add_argument('-s', '--sched', type=str, default=None)
    parser.add_argument('-y', '--sys', type=str, default=None)
    parser.add_argument('-i', '--agent_id', type=str, required=True)
    parser.add_argument('-r', '--round', type=int, default=150)
    parser.add_argument('-a', '--app_id', type=str, default=None)
    parser.add_argument('-f', '--func', type=str, required=True)
    args = parser.parse_args()
    f = args.func

    if f == 'plot_comp':
        plot_comparison(args)
    elif f == 'gen_q_data':
        generate_queues_data(args)

# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_sheduler/queue_q1
# /home/homidi/Desktop/research/projects/u-thr/scripts/../logs/20_agents/rr_scheduler/queue_q1/
