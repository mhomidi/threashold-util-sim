
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import unittest
from modules.applications.markov import MarkovApplication

class TestApplication(unittest.TestCase):
    
    def test_json_read(self):
        json_name = os.path.dirname(os.path.abspath(__file__)) + "/../json/unit_test_app.json"
        app = MarkovApplication()
        app.init_from_json(json_name)

        s1 = app.get_state_with_name("s1")
        s2 = app.get_state_with_name("s2")
        s3 = app.get_state_with_name("s3")

        assert(app.get_curr_state() is s1)
        assert(app.get_curr_state().get_utils() == [0.5,0.5,0.6,0.6])
        app.update_state()

        assert(app.get_curr_state() is s2 or app.get_curr_state() is s3)

if __name__ == "__main__":
    unittest.main()