import csv

from src.agents.no_reg_agent import NoRegretAgent

import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."


class Reporter:

    def write_weights(self, agent: NoRegretAgent) -> None:
        field = []
        for i in range(len(agent.weights)):
            field.append(str(i))

        data = []
        for i in range(len(agent.weights[0])):
            row = []
            for j in range(len(agent.weights)):
                row.append(agent.weights[j][i])
            data.append(row)

        file_name = root_dir + "/report_" + str(agent.get_id()) + ".csv"
        
        with open(file_name, "w") as f:
            writer = csv.writer(f)

            writer.writerow(field)
            writer.writerows(data)