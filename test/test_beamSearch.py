"""
Test tree:
Node = (node_idx, value)

                             (0 , 3)
                                |
                 -------------------------------
                |                               |
             (1 , 2)                         (2 , 4)
         ------- -------                 ------- -------
        |       |       |               |       |       |
     (3 , 5) (4 , 3) (5 , 7)         (6 , 2) (7 , 9) (8 , 5)
                                        |
                                     (9 , 12)
"""

from unittest import TestCase
from core.beam_search import BeamSearch


class TestBeamSearch(TestCase):
    class BS(BeamSearch):

        def expand(self, node, **kwargs):
            print kwargs['hola']

        def node_actions(self, node):
            values = [3, 2, 4, 5, 3, 7, 2, 9, 5]

            pass

        def evaluate(self, node):
            return node[1]

        def post_lookahead(self, node):
            return node[0], node[1] + 1


    def test_lookahead(self):
        self.fail()

    def test_run(self):
        self.fail()
