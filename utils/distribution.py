
import numpy as np

MAX_GNERATOR_NUM = 10


class Generator:

    def generate(self) -> float:
        raise NotImplementedError


class UniformGenerator(Generator):

    def generate(self) -> int:
        return np.random.randint(0, MAX_GNERATOR_NUM)


class GeometricGenerator(Generator):

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def generate(self) -> float:
        rand_num = 0.
        while True:
            rand_num = np.random.geometric(self.p)
            if rand_num < MAX_GNERATOR_NUM:
                break
        return rand_num


class PoissonGenerator(Generator):

    def __init__(self, lam=3) -> None:
        self.lam = lam
        super().__init__()

    def generate(self) -> float:
        rand_num = 0.
        while True:
            rand_num = np.random.poisson(self.lam)
            if rand_num < MAX_GNERATOR_NUM:
                break
        return rand_num
