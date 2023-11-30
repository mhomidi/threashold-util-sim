
from utils.pipe import Pipe
import config


class Dispatcher:
    def __init__(self) -> None:
        self.incoming_pipes = list()
        self.out_pipes = list()
        self.report = list()
        self.report_update = list()
        self.last_id = 0
        self.weights = list()

    def connect(self,
                incoming_pipe: Pipe,
                out_pipe: Pipe,
                weight: float = 1. / config.get('default_agent_num')
                ):
        self.incoming_pipes.append(incoming_pipe)
        self.out_pipes.append(out_pipe)
        self.report.append(None)
        self.report_update.append(False)
        self.weights.append(weight)
        incoming_pipe.set_id(self.last_id)
        out_pipe.set_id(self.last_id)
        self.last_id += 1

    def set_cluster_assignments(self, assignments: list) -> None:
        self.cluster_assignments = assignments

    def is_report_new(self):
        return sum(self.report_update) == len(self.report_update)

    def get_report(self) -> list:
        return self.report

    def get_weights(self) -> list:
        return self.weights

    def recieve_data(self) -> None:
        raise NotImplementedError()

    def send_data(self) -> None:
        raise NotImplementedError()

    def update_report(self, agent_id: int, data) -> None:
        self.report[agent_id] = data
        self.report_update[agent_id] = True
