from modules.scheduler.queue_sched.baseline.gandiva_fair import GandivaScheduler

import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

EPS = 1e-3


class GandivaTest(unittest.TestCase):

    def test_trading(self):
        sched = GandivaScheduler()

        # change config file for c_n = 2 and a_n = 3
        sched.report = [
            {'weight': 0.33, 'throughput': [0.1, 0.4]},
            {'weight': 0.33, 'throughput': [0.1, 0.6]},
            {'weight': 0.34, 'throughput': [0.1, 0.8]},
        ]

        sched.read_weights()
        sched.trade_resources()
        assert (sched.weights[0][0] == 1.)
        assert (sched.weights[1][0] == 0)
        assert (sched.weights[2][0] == 0)

        assert (sched.weights[0][1] > 0.1908 -
                EPS and sched.weights[0][1] < 0.1908 + EPS)
        assert (sched.weights[1][1] > 0.4125 -
                EPS and sched.weights[1][1] < 0.4125 + EPS)
        assert (sched.weights[2][1] > 0.3966 -
                EPS and sched.weights[2][1] < 0.3966 + EPS)


if __name__ == '__main__':
    unittest.main()
