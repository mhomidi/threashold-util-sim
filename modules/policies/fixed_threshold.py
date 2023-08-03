

from modules.policies import Policy
from config import config


class FixedThresholdPolicy(Policy):

    def __init__(self) -> None:
        super().__init__()

    def get_u_thr(self, data: list):
        return config.FIXED_THRESHOLD
    
    def train(self, reward: float, new_state_data: list):
        pass