

from modules.policies import Policy
import config


class FixedThresholdPolicy(Policy):

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def get_u_thr(self, data: list):
        return self.threshold
    
    def train(self, reward: float, new_state_data: list):
        pass