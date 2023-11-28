
from modules.applications import Application, State
from utils.distribution import *
import random


class QueueApplication(Application):

    def __init__(self, generator: Generator = UniformGenerator()) -> None:
        super().__init__()
        self.generator = generator
        self.init_state = State("init", [self.generator.generate()])
        self.curr_state = self.init_state
        self.length: int = self.init_state.val[0]
        self.speed_up = sorted([random.random()
                               for _ in range(config.get('cluster_num'))])

    def go_next_state(self) -> None:
        self.curr_state.set_val([self.generator.generate()])
        self.length += self.init_state.val[0]

    def get_length(self) -> int:
        return self.length

    def reduce_length(self, passed_job: int):
        self.length -= passed_job
        if self.length < 0:
            self.length = 0

    def get_throughput(self) -> list:
        return self.speed_up
