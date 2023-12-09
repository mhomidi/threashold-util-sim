from modules.applications import Application
import numpy as np


class MarkovApplication(Application):

    def __init__(self, utilities, transition_matrix, initial_index) -> None:
        super(MarkovApplication, self).__init__()
        self.utilities = utilities
        self.len_utilities = len(utilities)
        self.transition_matrix = transition_matrix
        self.state = self.utilities[initial_index]

    def update_state(self) -> None:
        self.state_history.append(self.state)
        current_index = self.utilities.index(self.state)
        trans = self.transition_matrix[current_index]
        self.state = np.random.choice(self.utilities, p=trans)
