from modules.applications import Application
import numpy as np


class MarkovApplication(Application):

    def __init__(self, utilities, transition_matrix, initial_index) -> None:
        super(MarkovApplication, self).__init__()
        self.utilities = utilities
        self.len_utilities = len(utilities)
        self.transition_matrix = transition_matrix
        self.curr_state = self.utilities[initial_index]

    def go_next_state(self) -> None:
        super().go_next_state()
        current_index = self.utilities.index(self.curr_state)
        trans = self.transition_matrix[current_index]
        self.curr_state = np.random.choice(self.utilities, p=trans)

    def get_utility(self):
        return self.curr_state
