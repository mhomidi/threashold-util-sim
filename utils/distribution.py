import numpy as np


class Generator:
    def __init__(self, rate):
        self.rate = rate

    def generate(self):
        raise NotImplementedError


class PoissonGenerator(Generator):

    def __init__(self, rate=3):
        super().__init__(rate)

    def generate(self):
        return np.random.poisson(self.rate)
