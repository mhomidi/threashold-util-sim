import argparse
import json
import os
import sys
import shutil

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)

from generate_plots import main as plot_main
from generate_avg import main as avg_main
from sys_setup import main as setup_main, get_agent_split_indices, get_agents_weights
from hetero_avg import plot_per_sched
from hetero_avg import plot_per_class
from plot_welfare import plot_welfare
from bar_plots import plot_average_plot_per_w

sched_args = {
    'g_fair': [1, 0, 1, 1],
    'themis': [1, 0, 2, 3],
    'mtf': [1, 0, 0, 0],
}

def do_sched(sched, config_file_path, config, queue_app_type, indices):
    print('====== ' + sched + ' =======')
    num_agents = config["num_agents"]
    app_type_id, app_type_sub_id, policy_id, scheduler_id = sched_args[sched]
    setup_main(config_file_path, app_type_id, app_type_sub_id,
                policy_id, scheduler_id, queue_app_type=queue_app_type)
    avg_main(num_agents, sched)
    plot_per_sched(num_agents, sched, indices)

def delete_folder(config, queue_app_type, sched=None):
    num_agents = config["num_agents"]
    
    if sched is None:
        path = f"{root}/logs/{num_agents}_agents/"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return
    app_type_id, app_type_sub_id, _, scheduler_id = sched_args[sched]
    app_type = config["app_types"][app_type_id]
    queue_app_type = config["queue_app_type"][queue_app_type]
    app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
    scheduler_type = config["scheduler_types"][scheduler_id]
    path = f"{root}/logs/{num_agents}_agents/{scheduler_type}/{app_type}_{app_sub_type}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sched', type=str, default=None)
    parser.add_argument('-p', '--plot', type=bool, default=False)
    parser.add_argument('-q', '--queue_app_type', type=str, default='wo_dd')
    args = parser.parse_args()
    sched = args.sched
    plot = args.plot
    queue_app_type = args.queue_app_type
    if queue_app_type == 'wo_dd':
        title = 'Average Accumulated Queue Length'
    elif queue_app_type == 'dd':
        title = 'Average Throughput'
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        
    num_agents = config["num_agents"]
    agents_per_class = config['agents_per_class']
    indices = get_agent_split_indices(num_agents, agents_per_class)
    classes = config['weight_of_classes']

    scheds = list(sched_args.keys())

    if sched:
        delete_folder(config, queue_app_type, sched)
        do_sched(sched, config_file_path, config, queue_app_type, indices)
    else:
        delete_folder(config, queue_app_type)
        for s in scheds:
            do_sched(s, config_file_path, config, queue_app_type, indices)

    if plot:
        print('====== Plot =======')
        plot_main(None, num_agents, title, True, scheds)
        for idx, c in enumerate(classes):
            plot_per_class(num_agents, indices, idx, c, scheds, title)
        ws = get_agents_weights(num_agents, indices, classes)
        ws = ws / ws.sum()
        ws = ws.reshape(1, num_agents)
        plot_welfare(num_agents, scheds, ws)
        plot_average_plot_per_w(num_agents, indices)
