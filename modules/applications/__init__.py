from __future__ import annotations


class Application:

    def __init__(self) -> None:
        self.states = list()
        self.curr_state = None

    def go_next_state(self) -> None:
        self.states.append(self.curr_state)

    def get_curr_state(self):
        return self.curr_state

    # This function should be called
    def get_utility(self):
        raise NotImplementedError
