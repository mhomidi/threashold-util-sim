from modules.applications.queue import QueueApplication
from modules.agents import QueueAgent
from utils.pipe import Pipe
from modules.dispatcher.queue import QueueDispatcher
from modules.scheduler.queue_sched.baseline.finish_time_fairness import FinishTimeFairnessScheduler
import config

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
        count += app.get_state().val[0]
        app.update_state()
        count += app.get_state().val[0]
        app.update_state()
        count += app.get_state().val[0]
        app.update_state()
        count += app.get_state().val[0]
        app.update_state()
        count += app.get_state().val[0]
        assert (app.get_length() == count)
        app.reduce_length(0.5)
        count -= 0.5
        assert (app.get_length() == count)

    def test_agent(self):
        atd = Pipe(0)
        dta = Pipe(0)

        app = QueueApplication()
        app.update_state()
        app.update_state()
        app.update_state()
        app.update_state()

        agent = QueueAgent(app)
        agent.connect(dta, atd)
        agent.send_data()

        data = atd.get()
        assert (data['q_length'] == app.get_length())

    def test_dispatcher(self):
        app1 = QueueApplication()
        app1.update_state()
        app1.update_state()
        app1.update_state()
        app1.update_state()

        app2 = QueueApplication()
        app2.update_state()
        app2.update_state()
        app2.update_state()
        app2.update_state()

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

    def test_finish_time_fairness_cheduler(self):
        sched = FinishTimeFairnessScheduler()
        a_num = config.get('default_agent_num')
        c_num = config.get('cluster_num')

        for i in range(a_num):
            dp = sched.get_dispatcher()

            app1 = QueueApplication()
            app1.update_state()
            app1.update_state()
            app1.update_state()
            app1.update_state()

            q11 = Pipe(0)
            q12 = Pipe(0)
            a1 = QueueAgent(app1)

            dp.connect(q11, q12)

            a1.connect(q12, q11)

            a1.send_data()

        dp.recieve_data()
        sched.set_report(dp.get_report())
        alloc = sched.schedule()
        assert(len(alloc) == c_num)


if __name__ == "__main__":
    unittest.main()
