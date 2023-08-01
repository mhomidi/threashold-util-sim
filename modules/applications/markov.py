

from modules.applications import Application, State
import utils
import random


class MarkovApplication(Application):

    def __init__(self) -> None:
        super(MarkovApplication, self).__init__()
        self.transitions = dict()

    def add_state(self, state: State) -> None:
        if self.init_state is None:
            self.init_state = state
            self.curr_state = state
        self.transitions[state] = dict()
        self.states.append(state)

    def add_transition(self, src: State, dest: State, prob: float) -> None:
        self.transitions[src][dest] = prob

    def go_next_state(self) -> None:
        trans = self.transitions[self.curr_state]
        rand = random.random()
        start_point = 0.0
        selected_state = None
        for state in trans:
            start_point += trans[state]
            if rand < start_point:
                selected_state = state
                break
        if selected_state is None:
            raise Exception()
        self.curr_state = selected_state

    def init_from_json(self, json_file: str) -> None:
        data = utils.get_json_data_from_file(json_file)
        for key in data:
            ru = State(key, data[key]["utils"])
            self.add_state(ru)
        for key in data:
            src = self.get_state_with_name(key)
            for state_name, prob in data[key]["transitions"].items():
                dest = self.get_state_with_name(state_name)
                self.add_transition(src, dest, prob)

    # def get_mean_util(self):
    #     mean = 0.
    #     for key in self.transitions:
    #         key: State
    #         u = sum(key.get_utils()) / len(key.get_utils())
    #         mean += u * key.get_prob()
    #     return mean