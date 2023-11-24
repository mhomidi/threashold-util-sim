

from modules.dispatcher.pref import Dispatcher
import config


class Scheduler:

    def __init__(self) -> None:
        self.report = None
        self.cluster_assignment = [
            -1 for _ in range(config.get('cluster_num'))]
        self.dispatcher = None
        self.assignment_history = list()

    def set_report(self, report: list) -> None:
        self.report = report

    def get_report(self) -> list:
        return self.report

    def get_dispatcher(self) -> Dispatcher:
        return self.dispatcher

    def get_cluster_assignments(self):
        return self.cluster_assignment

    def schedule(self) -> list:
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
