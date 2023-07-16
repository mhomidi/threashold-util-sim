
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from src.scheduler import *
import copy

import unittest

class TestScheduler(unittest.TestCase):

    def test_schedule(self):
        report = [
            [10, [2, 3, 5, 0, 1]],
            [8, [2, 7, 0, 1]],
            [9, [5, 3, 8]],
        ]

        sched = Scheduler()
        sched.set_report(copy.deepcopy(report))
        alloc = sched.schedule()

        assert(alloc == [0, 0, 0, 0, -1, 2, -1, 1, 2, -1])
        assert(sched.report[0][0] == 6)
        assert(sched.report[1][0] == 7)
        assert(sched.report[2][0] == 7)


if __name__ == "__main__":
    unittest.main()