
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import unittest
from modules.markov import application

class TestApplication(unittest.TestCase):
    
    def test_json_read(self):
        json_name = os.path.dirname(os.path.abspath(__file__)) + "/../json/unit_test_app.json"
        app = application.Application()
        app.init_from_json(json_name)

        s1 = app.get_state_with_name("s1")
        s2 = app.get_state_with_name("s2")
        s3 = app.get_state_with_name("s3")

        assert(app.get_curr_state() is s1)
        assert(app.get_curr_state().get_val() == [0.5,0.5,0.6,0.6])
        app.go_next_state()

        assert(app.get_curr_state() is s2 or app.get_curr_state() is s3)

    def test_get_mean(self):
        json_name = os.path.dirname(os.path.abspath(__file__)) + "/../json/unit_test_app.json"
        app = application.Application()
        app.init_from_json(json_name)

        s1 = app.get_state_with_name("s1")
        s2 = app.get_state_with_name("s2")
        s3 = app.get_state_with_name("s3")

        mean_s1 = sum(s1.get_val()) / len(s1.get_val())
        mean_s2 = sum(s2.get_val()) / len(s2.get_val())
        mean_s3 = sum(s3.get_val()) / len(s3.get_val())

        mean = s1.get_prob() * mean_s1 + s2.get_prob() * mean_s2 + s3.get_prob() * mean_s3
        assert(mean == app.get_mean_util())

if __name__ == "__main__":
    unittest.main()