
import sys
import os
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/../"

import pandas as pd

N = 5

if __name__ == "__main__":
    data = [[] for _ in range(N)]
    for main_agent in range(N):
        utils = pd.read_csv(root_dir + 'a' + str(main_agent) + '_uhist.csv')
        utils = utils[4999:].to_numpy(np.float32)
        assignment = pd.read_csv(root_dir + 'report_assignments.csv')
        assignment = assignment[5000:].to_numpy(np.int32)

        for i in range(N):
            util = np.sum(utils[assignment == i]) / 5000
            data[main_agent].append(util)
    
    df = pd.DataFrame(data)
    df.to_csv('ef.csv')

        
