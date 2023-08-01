
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from modules.scheduler.most_token_first import *
import copy

import unittest

class TestScheduler(unittest.TestCase):

    def test_schedule(self):
        report = [
            [10, [2, 3, 5, 0, 1]],
            [8, [2, 7, 0, 1]],
            [9, [5, 3, 8]],
        ]

        sched = MostTokenFirstScheduler()
        sched.set_report(copy.deepcopy(report))
        alloc = sched.schedule()

        assert(alloc == [0, 0, 0, 0, -1, 2, -1, 1, 2, -1])

        new_b = sched.get_new_budgets()
        assert(new_b == [6, 7, 7])


if __name__ == "__main__":
    unittest.main()