from modules.applications.queue import QueueApplication
from modules.agents import QueueAgent
from utils.pipe import Pipe
from modules.dispatcher.queue import QueueDispatcher

import unittest
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


class QueueTest(unittest.TestCase):

    def test_application(self):
        random.seed("test")
        count = 0.
        app = QueueApplication()
        count += app.get_curr_state().val[0]
        app.go_next_state()
        count += app.get_curr_state().val[0]
        app.go_next_state()
        count += app.get_curr_state().val[0]
        app.go_next_state()
        count += app.get_curr_state().val[0]
        app.go_next_state()
        count += app.get_curr_state().val[0]
        assert (app.get_length() == count)
        app.reduce_length(0.5)
        count -= 0.5
        assert (app.get_length() == count)

    def test_agent(self):
        atd = Pipe(0)
        dta = Pipe(0)

        app = QueueApplication()
        app.go_next_state()
        app.go_next_state()
        app.go_next_state()
        app.go_next_state()

        agent = QueueAgent(app)
        agent.connect(dta, atd)
        agent.send_data()

        data = atd.get()
        assert (data['q_length'] == app.get_length())

    def test_dispatcher(self):
        app1 = QueueApplication()
        app1.go_next_state()
        app1.go_next_state()
        app1.go_next_state()
        app1.go_next_state()

        app2 = QueueApplication()
        app2.go_next_state()
        app2.go_next_state()
        app2.go_next_state()
        app2.go_next_state()

        q11 = Pipe(0)
        q12 = Pipe(0)
        q21 = Pipe(1)
        q22 = Pipe(1)
        a1 = QueueAgent(app1)
        a2 = QueueAgent(app2)

        a1.connect(q12, q11)
        a2.connect(q22, q21)

        dp = QueueDispatcher()
        dp.connect(q11, q12)
        dp.connect(q21, q22)

        a1.send_data()
        a2.send_data()

        dp.recieve_data()
        report = dp.get_report()

        assert (report[0]['throughput'] == app1.get_throughput())
        assert (report[0]['q_length'] == app1.get_length())

        assert (report[1]['throughput'] == app2.get_throughput())
        assert (report[1]['q_length'] == app2.get_length())


if __name__ == "__main__":
    unittest.main()
