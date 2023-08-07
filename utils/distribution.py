
import numpy as np
import config

class Generator:

    def generate(self) -> float:
        raise NotImplementedError()
    
    
class UniformGenerator(Generator):
    
    def generate(self) -> float:
        return np.random.randint(0, config.get('util_interval')) / config.get('util_interval')


class GeometricGenerator(Generator):

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    def generate(self) -> float:
        rand_num = 0.
        while True:
            rand_num = np.random.geometric(self.p)
            if rand_num < config.get('util_interval'):
                break
        return rand_num / config.get('util_interval')
    
class PoissonGenerator(Generator):

    def __init__(self, lam=3) -> None:
        self.lam = lam
        super().__init__()

    def generate(self) -> float:
        rand_num = 0.
        while True:
            rand_num = np.random.poisson(self.lam)
            if rand_num < config.get('util_interval'):
                break
        return rand_num / config.get('util_interval')


class UtilityGenerator:

    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def generate_utilities(self) -> list():
        utils = []
        for _ in range(config.get('cluster_num')):
            utils.append(self.generator.generate())
        return utils