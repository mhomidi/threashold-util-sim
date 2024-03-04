
import numpy as np

COLORS = ['purple', 'orange', 'green', 'red', 'yellow', 'gray']

SCHED_TITLES = {
    'themis': 'Themis',    
    'g_fair': 'Gandiva_fair',    
    'l_dice': 'L-Dice',
    's_dice': 'S-Dice',
    'm_dice': 'M-Dice',
}

UTILS = {
    '40': '40%',    
    '50': '50%',    
    '60': '60%',    
    '70': '70%',    
    '80': '80%'   
}

DEADLINES = [2, 5, 10, 15, 20]

WEIGHT_TEXTS = ['1', '12', '124', '128']
INDICES = [[], [10], [10, 15], [10, 15]]


def get_agents_weights(num_agents, agent_split_indices, weight_of_classes):
    agent_weights = np.ones(num_agents)
    if agent_split_indices is None or len(agent_split_indices) < 2:
        return agent_weights 
    assert len(agent_split_indices) + 1 == len(weight_of_classes)
    ids_weight_classes = np.split(
        np.arange(0, num_agents), agent_split_indices)
    for wc, wpc in zip(ids_weight_classes, weight_of_classes):
        agent_weights[wc] *= wpc
    return agent_weights

def get_classes_from_weight_text(weight):
    weight = int(weight)
    classes = []
    while weight > 0:
        classes.append(weight % 10)
        weight /= 10
        weight = int(weight)
    return classes[::-1]
        