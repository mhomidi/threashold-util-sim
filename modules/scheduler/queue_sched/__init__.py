
from modules.scheduler import Scheduler
from modules.dispatcher.queue import QueueDispatcher
import config
import numpy as np


class QueueBaseScheduler(Scheduler):

    def __init__(self) -> None:
        super().__init__()
        self.dispatcher = QueueDispatcher()

    def run(self):
        for episode in range(config.get('episodes')):
            self.dispatcher.recieve_data()
            self.set_report(self.dispatcher.get_report())
            allocation = self.schedule()
            self.dispatcher.set_cluster_assignments(allocation)
            self.dispatcher.set_passed_jobs(self.get_passed_jobs(allocation))
            self.assignment_history.append(allocation)
            self.dispatcher.send_data()
            if episode % 500 == 499:
                print("episode {e} done".format(e=episode + 1))

    def get_passed_jobs(self, allocation: list) -> list:
        passed_jobs = list()
        for i in range(config.get(config.AGENT_NUM)):
            q_length = self.report[i]['q_length']
            throughput = np.array(self.report[i]['throughput'])[
                np.array(allocation) == i].sum()
            passed_jobs.append(min(q_length, throughput))
        return passed_jobs
