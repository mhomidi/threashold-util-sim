from __future__ import annotations


class Application:

    def __init__(self) -> None:
        self.state_history = list()
        self.curr_state = None

    def update_state(self) -> None:
        self.state_history.append(self.curr_state)

    def get_curr_state(self):
        return self.curr_state
