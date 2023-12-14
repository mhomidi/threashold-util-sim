import numpy as np


class Generator:

    def generate(self):
        raise NotImplementedError


class PoissonGenerator(Generator):

    def __init__(self, lam=3):
        self.lam = lam
        super().__init__()

    def generate(self):
        return np.random.poisson(self.lam)
