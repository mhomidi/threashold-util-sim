import csv

from modules.agents import PrefAgent
from modules.scheduler.pref_sched import PrefScheduler
from modules import agents
import numpy as np

import os

root_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."

# Report param
UTILITY_DATA_TYPE = 0
TOKEN_DATA_TYPE = 1
TOKEN_DIST_TYPE = 2
ASSIGNMENT_TYPE = 3
UTILS_HISTORY = 4


class Report:

    def __init__(self) -> None:
        self.rewards = list()
        self.tokens = list()
        self.agents = list()
        self.token_dists = list()
        self.scheduler = None
        self.utils_histories = list()

    def add_agent(self, agent: PrefAgent):
        self.agents.append(agent)
        self.utils_histories.append(list())

    def set_scheduler(self, sched: PrefScheduler) -> None:
        self.scheduler = sched

    def generate_tokens_row(self):
        for agent in self.agents:
            agent: PrefAgent
            self.tokens.append(agent.budgets_history)

    def generate_token_distributions_row(self):
        for agent in self.agents:
            agent: PrefAgent
            self.token_dists.append(agent.token_dist)

    def generate_rewards_row(self):
        for agent in self.agents:
            agent: PrefAgent
            self.rewards.append(agent.rewards_history)

    def generate_utils_histories(self):
        for agent in self.agents:
            agent: PrefAgent
            self.utils_histories[agent.id] = agent.utils_history

    def write_data(self, data_type=UTILITY_DATA_TYPE):
        field = list()
        for agent in self.agents:
            agent: PrefAgent
            field.append(agent.id)

        file_name = None
        data = None

        if data_type == UTILITY_DATA_TYPE:
            file_name = root_dir + "/report_utils.csv"
            data = self.rewards
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

    def write_multiple_data(self, data_type=UTILS_HISTORY):
        file_names = list()
        data = list()
        iteration = None

        if data_type == UTILS_HISTORY:
            iteration = self.agents
            for agent in iteration:
                agent: PrefAgent
                data.append(self.utils_histories[agent.id])
                file_names.append(root_dir + "/a" +
                                  str(agent.id) + "_uhist.csv")
        else:
            raise Exception()

        for i, d in enumerate(data):
            csv_data = np.array(d)
            file_name = file_names[i]
            with open(file_name, "w") as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

    def prepare_report(self):
        self.generate_rewards_row()
        self.write_data(UTILITY_DATA_TYPE)
        self.write_data(ASSIGNMENT_TYPE)

        if isinstance(self.agents[0], agents.PrefAgent):
            self.generate_tokens_row()
            self.generate_utils_histories()
            self.write_data(TOKEN_DATA_TYPE)
            self.write_multiple_data(UTILS_HISTORY)
