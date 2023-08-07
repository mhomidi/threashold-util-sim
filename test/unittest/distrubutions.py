import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from utils.distribution import *
import unittest
import config


class TestDistribution(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.nums = [i / float(config.get('util_interval')) for i in range(config.get('util_interval'))]
        self.n = 100

    def test_uniform(self):
        for _ in range(self.n):
            num = UniformGenerator().generate()
            assert(num in self.nums)

    def test_poisson(self):
        for _ in range(self.n):
            num = PoissonGenerator().generate()
            assert(num in self.nums)

    def test_geometric(self):
        for _ in range(self.n):
            num = GeometricGenerator().generate()
            assert(num in self.nums)


if __name__ == "__main__":
    unittest.main()