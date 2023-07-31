import csv

from modules.agents.no_reg_agent import NoRegretAgent
from modules.agents import Agent

import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

# Report param
UTILITY_DATA_TYPE = 0
TOKEN_DATA_TYPE = 1


class Report:

    def __init__(self) -> None:
        self.utilities = list()
        self.tokens = list()
        self.agents = list()

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def generate_tokens_row(self):
        tokens = []
        for agent in self.agents:
            tokens.append(agent.get_budget())
        self.tokens.append(tokens)

    def generate_utilities_row(self):
        utilities = []
        for agent in self.agents:
            agent: Agent
            utilities.append(agent.get_round_utility())
        self.utilities.append(utilities)

    def write_data(self, data_type=UTILITY_DATA_TYPE):
        field = list()
        for agent in self.agents:
            field.append(agent.get_id())

        file_name = None
        data = None
        
        if data_type == UTILITY_DATA_TYPE:
            file_name = root_dir + "/report_utils.csv"
            data = self.utilities
        elif data_type == TOKEN_DATA_TYPE:
            file_name = root_dir + "/report_tokens.csv"
            data = self.tokens
        else:
            raise Exception()

        with open(file_name, "w") as f:
            writer = csv.writer(f)

            writer.writerow(field)
            writer.writerows(data)

class NRAgentReporter(Report):

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