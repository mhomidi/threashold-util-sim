

from modules.policies import Policy
import config


class FixedThresholdPolicy(Policy):

    def __init__(self) -> None:
        super().__init__()

    def get_u_thr(self, data: list):
        return config.get('fixed_threshold')
    
    def train(self, reward: float, new_state_data: list):
        pass