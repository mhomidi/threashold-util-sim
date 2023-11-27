
from modules.scheduler import Scheduler
from modules.dispatcher.queue import QueueDispatcher
import config


class QueueBaseScheduler(Scheduler):

    def __init__(self) -> None:
        super().__init__()
        self.dispatcher = QueueDispatcher()

    def run(self):
        for episode in range(config.get('episodes')):
            self.dispatcher.recieve_data()
            self.set_report(self.dispatcher.get_report())
            self.assignment_history.append(self.schedule())
            self.dispatcher.send_data()
            if episode % 500 == 499:
                print("episode {e} done".format(e=episode + 1))
