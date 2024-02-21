
import numpy as np

COLORS = ['purple', 'orange', 'green', 'red', 'yellow']

SCHED_TITLES = {
    'g_fair': 'Gandiva_fair',    
    'themis': 'Themis',    
    'wrr': 'WRR',
    'ceei': 'CEEI',
    'mtf': 'MTF',    
}

def get_agents_weights(num_agents, agent_split_indices, weight_of_classes):
    agent_weights = np.ones(num_agents)
    assert len(agent_split_indices) + 1 == len(weight_of_classes)
    ids_weight_classes = np.split(
        np.arange(0, num_agents), agent_split_indices)
    for wc, wpc in zip(ids_weight_classes, weight_of_classes):
        agent_weights[wc] *= wpc
    return agent_weights