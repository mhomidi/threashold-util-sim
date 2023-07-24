
import numpy as np
from config import config

class MeanGenerator:

    def generate_mean(self):
        raise NotImplementedError()
    
    def get_mean_length(self) -> float:
        self.mean = self.generate_mean()
        if self.mean >= 1. or self.mean < 0:
            raise Exception()
        random_lenght = np.random.random() * 0.5
        self.length = min(1. - self.mean, self.mean, random_lenght)
        return self.mean, self.length
    
    
class UniformMeanGenerator(MeanGenerator):

    def generate_mean(self):
        return np.random.random()
    

class GeometricMeanGenerator(MeanGenerator):

    def __init__(self, p=0.5, maximum=config.DIST_SAMPLE) -> None:
        self.p = p
        self.maximum = maximum
        super().__init__()

    def generate_mean(self) -> float:
        rand_num = 0.
        while True:
            rand_num = np.random.geometric(self.p)
            if rand_num < self.maximum:
                break
        return rand_num / self.maximum
    
class PoissonMeanGenerator(MeanGenerator):

    def __init__(self, lam=3, maximum=config.DIST_SAMPLE) -> None:
        self.lam = lam
        self.maximum = maximum
        super().__init__()

    def generate_mean(self):
        rand_num = 0.
        while True:
            rand_num = np.random.poisson(self.lam)
            if rand_num < self.maximum and rand_num > 0:
                break
        return rand_num / self.maximum


class UtilityGenerator:

    def __init__(self, mean_gen: MeanGenerator) -> None:
        self.mean_gen = mean_gen

    def get_utilities(self) -> list():
        self.mean, self.length = self.mean_gen.get_mean_length()
        utils = np.random.uniform(
            low=(self.mean - self.length/2),
            high=(self.mean + self.length/2),
            size=config.CLUSTERS_NUM
            )
        return utils.tolist()