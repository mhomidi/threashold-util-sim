import argparse
import json
import os
import sys
import shutil

root = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(root)

from generate_plots import main as plot_main
from generate_avg import main as avg_main
from sys_setup import main as setup_main, get_agent_split_indices
from script_utils import get_agents_weights 
from hetero_avg import plot_per_sched
from hetero_avg import plot_per_class
from plot_welfare import plot_welfare
from bar_plots import plot_average_plot_per_w

sched_args = {
    'tant': [0, 2, 2],
    'ss': [0, 0, 0],
    'l_dice': [0, 1, 1],
    's_dice': [0, 4, 4],
    'm_dice': [0, 3, 3]
}

def do_sched(sched, config_file_path, config, deadline_type, indices, app_type_sub_id):
    num_agents = config["num_agents"]
    c_num = config["num_nodes"]
    queue_util = config["queue_util"]
    deadline = config['deadline']
    classes = config['weight_of_classes']
    weights_text = "".join([str(item) for item in classes])

    print('====== ' + sched + ' =======')
    num_agents = config["num_agents"]
    app_type_id, policy_id, scheduler_id = sched_args[sched]
    setup_main(config_file_path, app_type_id, app_type_sub_id,
                policy_id, scheduler_id, deadline_type=deadline_type)
    avg_main(num_agents, sched, c_num, queue_util, weights_text, dd=deadline, app_sub_id=app_type_sub_id)
    plot_per_sched(num_agents, sched, indices, c_num, queue_util, weights_text, dd=deadline, app_sub_id=app_type_sub_id)

def recreate_folder(config, deadline_type, sched=None):
    num_agents = config["num_agents"]
    c_num = config["num_nodes"]
    util = config["queue_util"]
    classes = config['weight_of_classes']
    deadline = config['deadline']
    weights_text = "".join([str(item) for item in classes])
    
    if sched is None:
        path = f"{root}/logs/{num_agents}-{c_num}-{util}util-{weights_text}/"
        if deadline_type == 'dd':
            path = f"{root}/logs/{num_agents}-{c_num}-{util}util-{weights_text}-dd{deadline}/"
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        return
    app_type_id, _, scheduler_id = sched_args[sched]
    app_type = config["app_types"][app_type_id]
    app_sub_type = config["app_sub_types"][app_type][app_type_sub_id]
    scheduler_type = config["scheduler_types"][scheduler_id]
    path = f"{root}/logs/{num_agents}-{c_num}-{util}util-{weights_text}/{scheduler_type}/{app_type}_{app_sub_type}"
    if deadline_type == 'dd':
        path = f"{root}/logs/{num_agents}-{c_num}-{util}util-{weights_text}-dd{deadline}/{scheduler_type}/{app_type}_{app_sub_type}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sched', type=str, default=None)
    parser.add_argument('-p', '--plot', type=bool, default=False)
    parser.add_argument('-r', '--should_run', type=bool, default=False)
    parser.add_argument('-d', '--deadline_type', type=str, default='dd')
    parser.add_argument('-q', '--app_type_sub_id', type=int, default=0)
    args = parser.parse_args()
    sched = args.sched
    should_run = args.should_run
    plot = args.plot
    deadline_type = args.deadline_type
    app_type_sub_id = args.app_type_sub_id
    config_file_path = root + "/config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
        

    deadline = None
    if deadline_type == 'wo_dd':
        title = 'Average Accumulated Queue Length'
    elif deadline_type == 'dd':
        title = 'Average Throughput'
        deadline = config["deadline"]
    num_agents = config["num_agents"]
    num_nodes = config["num_nodes"]
    queue_util = config['queue_util']
    agents_per_class = config['agents_per_class']
    indices = get_agent_split_indices(num_agents, agents_per_class)
    classes = config['weight_of_classes']

    scheds = list(sched_args.keys())

    if should_run:
        if sched:
            recreate_folder(config, deadline_type, sched)
            do_sched(sched, config_file_path, config, deadline_type, indices, app_type_sub_id)
        else:
            recreate_folder(config, deadline_type)
            for s in scheds:
                do_sched(s, config_file_path, config, deadline_type, indices, app_type_sub_id)

    if plot:
        print('====== Plot =======')
        weight_text = ""
        weight_text = weight_text.join([str(int(i)) for i in classes])
        ws = get_agents_weights(num_agents, indices, classes)
        ws = ws / ws.sum()
        ws = ws.reshape(1, num_agents)
        plot_main(num_agents, title, True, num_nodes, queue_util, weight_text, dd=deadline, scheds=scheds, app_sub_id=app_type_sub_id)
        for idx, c in enumerate(classes):
            plot_per_class(num_agents, indices, idx, c, scheds, title, num_nodes, queue_util, weight_text, dd=deadline, app_sub_id=app_type_sub_id)
        plot_welfare(num_agents, scheds, ws, num_nodes, queue_util, weight_text, dd=deadline, app_type_id=app_type_sub_id)
        plot_average_plot_per_w(num_agents, indices, num_nodes, queue_util, weight_text, dd=deadline, app_type_id=app_type_sub_id)
