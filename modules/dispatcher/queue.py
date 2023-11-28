
from modules.dispatcher import Dispatcher
from utils.pipe import Pipe
import time


class QueueDispatcher(Dispatcher):

    def set_passed_jobs(self, passed_jobs: list) -> None:
        self.passed_jobs = passed_jobs

    def send_data(self) -> None:
        for id, queue in enumerate(self.out_pipes):
            queue: Pipe
            data = dict()
            data["assignments"] = self.cluster_assignments
            data['passed'] = self.passed_jobs[id]
            queue.put(data=data)

    def recieve_data(self) -> None:
        while not self.is_report_new():
            time.sleep(0.0001)
            for pipe in self.incoming_pipes:
                pipe: Pipe
                if pipe.is_empty():
                    continue
                data = pipe.get()
                if data:
                    self.update_report(data['id'], data)
        self.report_update = [False for _ in range(len(self.incoming_pipes))]
