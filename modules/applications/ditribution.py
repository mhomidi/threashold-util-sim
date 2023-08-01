
from modules.applications import Application, State
from utils.distribution import Generator, UniformGenerator, UtilityGenerator


class DistributionApplication(Application):

    def __init__(self, generator: Generator = UniformGenerator()) -> None:
        super().__init__()
        self.utility_generator = UtilityGenerator(generator)
        self.init_state = State("init", self.utility_generator.generate_utilities())
        self.curr_state = self.init_state
        self.states = [self.curr_state]

    def go_next_state(self) -> None:
        self.curr_state = State("next", self.utility_generator.generate_utilities())
        self.states = [self.curr_state]
