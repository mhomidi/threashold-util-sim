

from modules import markov
import utils


class RoundUtility(markov.State):

    def __init__(self, name: str, val: list) -> None:
        super().__init__(name, val)


class Application(markov.MarkovChain):

    def init_from_json(self, json_file: str) -> None:
        data = utils.get_json_data_from_file(json_file)
        for key in data:
            ru = RoundUtility(key, data[key]["utils"])
            self.add_state(ru)
        for key in data:
            src = self.get_state_with_name(key)
            for state_name, prob in data[key]["transitions"].items():
                dest = self.get_state_with_name(state_name)
                self.add_transition(src, dest, prob)
