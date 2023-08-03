
class Policy:

    def __init__(self) -> None:
        self.state = None
        self.prev_state = None

    def get_u_thr(self, data: list):
        raise NotImplementedError()

    def train(self, reward: float, new_state_data: list):
        raise NotImplementedError()