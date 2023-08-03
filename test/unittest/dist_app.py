
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import unittest
from modules.applications.distribution import DistributionApplication


class TestApplication(unittest.TestCase):
    
   def test_go_to_next_state(self):
       app = DistributionApplication()
       utils = app.get_curr_state().get_utils()
       app.go_next_state()
       assert(utils != app.get_curr_state().get_utils())


if __name__ == "__main__":
    unittest.main()