import argparse
import json
import os
import sys

root = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(root)

from generate_plots import main as plot_main
from generate_avg import main as avg_main
from sys_setup import main as setup_main

sched_args = {
    'mtf': [1, 0, 0, 0],
    'g_fair': [1, 0, 1, 1],
    'themis': [1, 0, 2, 3],
    # 'proportional': [1, 0, 1, 4],
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sched', type=str, default=None)
    parser.add_argument('-p', '--plot', type=bool, default=False)
    parser.add_argument('-t', '--title', type=str, default='No Title')
    parser.add_argument('-q', '--queue_app_type', type=str, default='wo_dd')
    args = parser.parse_args()
    sched = args.sched
    plot = args.plot
    title = args.title
    queue_app_type = args.queue_app_type
    config_file_path = root + "config/sys_config_default.json"
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    num_agents = config["num_agents"]

    schedulers = sched_args.keys()

    if sched:
        print('====== ' + sched + ' =======')
        app_type_id, app_type_sub_id, policy_id, scheduler_id = sched_args[sched]
        setup_main(config_file_path, app_type_id, app_type_sub_id,
                   policy_id, scheduler_id, queue_app_type=queue_app_type)
        avg_main(num_agents, sched)

    else:
        for sched in schedulers:
            print('====== ' + sched + ' =======')
            app_type_id, app_type_sub_id, policy_id, scheduler_id = sched_args[sched]
            setup_main(config_file_path, app_type_id, app_type_sub_id,
                       policy_id, scheduler_id, queue_app_type=queue_app_type)
            avg_main(num_agents, sched)

    if plot:
        print('====== Plot =======')
        plot_main(None, num_agents, title, True)
