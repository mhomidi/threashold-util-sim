
from __future__ import annotations
import random


class State:

    def __init__(self, name: str, val) -> None:
        self.name = name
        self.val = val

    def get_val(self) -> float:
        return self.val

    def set_val(self, val: float) -> None:
        self.val = val


class MarkovChain:

    def __init__(self) -> None:
        self.init_state = None
        self.curr_state = None
        self.transitions = dict()

    def add_state(self, state: State) -> None:
        if self.init_state is None:
            self.init_state = state
            self.curr_state = state
        self.transitions[state] = dict()

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

    def get_curr_state(self) -> State:
        return self.curr_state

    def reset(self) -> None:
        self.curr_state = self.init_state

    def get_state_with_name(self, name: str) -> State:
        for state in self.transitions:
            if state.name == name:
                return state
        return None
