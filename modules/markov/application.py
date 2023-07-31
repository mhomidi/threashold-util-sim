

from modules import markov
import utils


class RoundUtility(markov.State):

    def __init__(self, name: str, val: list, prob: float) -> None:
        super().__init__(name, val, prob)


class Application(markov.MarkovChain):

    def init_from_json(self, json_file: str) -> None:
        data = utils.get_json_data_from_file(json_file)
        for key in data:
            ru = RoundUtility(key, data[key]["utils"], data[key]["prob"])
            self.add_state(ru)
        for key in data:
            src = self.get_state_with_name(key)
            for state_name, prob in data[key]["transitions"].items():
                dest = self.get_state_with_name(state_name)
                self.add_transition(src, dest, prob)

    def get_mean_util(self):
        mean = 0.
        for key in self.transitions:
            key: RoundUtility
            u = sum(key.get_val()) / len(key.get_val())
            mean += u * key.get_prob()
        return mean