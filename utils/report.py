import csv

from modules.agents import Agent
from modules.scheduler import Scheduler
import numpy as np

import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

# Report param
UTILITY_DATA_TYPE = 0
TOKEN_DATA_TYPE = 1
TOKEN_DIST_TYPE = 2
ASSIGNMENT_TYPE = 3


class Report:

    def __init__(self) -> None:
        self.utilities = list()
        self.tokens = list()
        self.agents = list()
        self.token_dists = list()
        self.scheduler = None

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def set_scheduler(self, sched: Scheduler) -> None:
        self.scheduler = sched

    def generate_tokens_row(self):
        for agent in self.agents:
            agent: Agent
            self.tokens.append(agent.budgets_history)

    def generate_token_distributions_row(self):
        for agent in self.agents:
            agent: Agent
            self.token_dists.append(agent.token_dist)

    def generate_utilities_row(self):
        for agent in self.agents:
            agent: Agent
            self.utilities.append(agent.utils_history)

    def write_data(self, data_type=UTILITY_DATA_TYPE):
        field = list()
        for agent in self.agents:
            agent: Agent
            field.append(agent.id)

        file_name = None
        data = None
        
        if data_type == UTILITY_DATA_TYPE:
            file_name = root_dir + "/report_utils.csv"
            data = self.utilities
        elif data_type == TOKEN_DATA_TYPE:
            file_name = root_dir + "/report_tokens.csv"
            data = self.tokens
        elif data_type == TOKEN_DIST_TYPE:
            file_name = root_dir + "/report_token_dists.csv"
            data = self.token_dists
        elif data_type == ASSIGNMENT_TYPE:
            file_name = root_dir + "/report_assignments.csv"
            data = self.scheduler.assignment_history
            data = np.array(data).T.tolist()
        else:
            raise Exception()

        data = np.array(data).T.tolist()
        with open(file_name, "w") as f:
            writer = csv.writer(f)

            writer.writerow(field)
            writer.writerows(data)