
from __future__ import annotations
import random


class State:

    def __init__(self, name: str, val: list) -> None:
        self.name = name
        self.val = val

    def get_utils(self) -> list:
        return self.val

    def set_val(self, val: list) -> None:
        self.val = val


class Application:

    def __init__(self) -> None:
        self.states = list()
        self.init_state = None
        self.curr_state = None

    def go_next_state(self) -> None:
        raise NotImplementedError()

    def get_curr_state(self) -> State:
        return self.curr_state

    def reset(self) -> None:
        self.curr_state = self.init_state

    def get_state_with_name(self, name: str) -> State:
        for state in self.states:
            state: State
            if state.name == name:
                return state
        return None
