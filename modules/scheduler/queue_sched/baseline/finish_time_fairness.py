
from modules.scheduler.queue_sched import QueueBaseScheduler

class FinishTimeFairnessScheduler(QueueBaseScheduler):

    def schedule(self) -> list:
        pass