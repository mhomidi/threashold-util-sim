
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

import unittest
from modules import markov


class TestApplication(unittest.TestCase):
    
    def test_add_state(self):
        chain = markov.MarkovChain()
        s1 = markov.State('s1', 1)
        s2 = markov.State('s2', 2)
        s3 = markov.State('s3', 3)
        chain.add_state(s1)
        chain.add_state(s2)
        chain.add_state(s3)

        assert(s1 in chain.transitions)
        assert(s2 in chain.transitions)
        assert(s3 in chain.transitions)

        assert(chain.get_curr_state() == s1)

    def test_transition(self):
        chain = markov.MarkovChain()
        s1 = markov.State('s1', 1)
        s2 = markov.State('s2', 2)
        s3 = markov.State('s3', 3)
        chain.add_state(s1)
        chain.add_state(s2)
        chain.add_state(s3)

        chain.add_transition(s1, s2, 0.5)
        chain.add_transition(s1, s3, 0.5)
        chain.add_transition(s2, s3, 0.5)
        chain.add_transition(s2, s1, 0.5)
        chain.add_transition(s3, s1, 0.5)
        chain.add_transition(s3, s2, 0.5)

        assert(chain.get_curr_state() == s1)
        chain.go_next_state()
        assert(chain.get_curr_state() == s2 or chain.get_curr_state() == s3)
        if chain.get_curr_state() is s2:
            chain.go_next_state()
            assert(chain.get_curr_state() == s1 or chain.get_curr_state() == s3)
        elif chain.get_curr_state() is s3:
            chain.go_next_state()
            assert(chain.get_curr_state() == s1 or chain.get_curr_state() == s2)


if __name__ == "__main__":
    unittest.main()