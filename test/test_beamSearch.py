"""
Test tree:
Node = (node_idx, value)

                             (0 , 1)
                                |
                 -------------------------------
                |                               |
             (1 , 2)                         (2 , 4)
         ------- -------                 ------- -------
        |       |       |               |       |       |
     (3 , 6) (4 , 3) (5 , 7)         (6 , 2) (7 , 9) (8 , 5)
                |       |               |
             (9 , 14)(10 , 10)       (11 , 12)
"""

from unittest import TestCase
from core.beam_search import BeamSearch
import numpy as np


class TestBeamSearch(TestCase):
    class BS(BeamSearch):
        def __init__(self):
            self.children = {
                0: [(1, 2), (2, 4)],
                1: [(3, 6), (4, 3), (5, 7)],
                2: [(6, 2), (7, 9), (8, 5)],
                3: [],
                4: [(9, 14)],
                5: [(10, 10)],
                6: [(11, 12)],
                7: [],
                8: [],
            }
            super(TestBeamSearch.BS, self).__init__(debug=True)

        def node_actions(self, node):
            actions = []

            def action(n, children, children_idx):
                return children[n[0]][children_idx]

            for c_idx in range(len(self.children[node[0]])):
                actions.append((action, {'children_idx': c_idx, 'children': self.children}))
            return actions

        def evaluate(self, node):
            return node[1]

        def post_lookahead(self, node):
            return node[0], node[1] + 1

    def test_lookahead(self):
        bs = self.BS()
        nodes, parents, scores, actions, bw = bs.lookahead((0, 1), k=3, depth=2)
        self.assertTrue(nodes == [(0, 1), (2, 4), (1, 2), (7, 9), (5, 7), (3, 6)])
        self.assertTrue(parents == [0, 0, 0, 1, 2, 2])
        self.assertTrue(scores == [-1, 4, 2, 9, 7, 6])

        nodes, parents, scores, actions, bw = bs.lookahead((0, 1), k=3, depth=3)
        self.assertTrue(nodes == [(0, 1), (2, 4), (1, 2), (7, 9), (5, 7), (3, 6), (10, 10)])
        self.assertTrue(parents == [0, 0, 0, 1, 2, 2, 4])
        self.assertTrue(scores == [-1, 4, 2, 9, 7, 6, 10])

    def test_run(self):
        bs = self.BS()
        nodes, scores, actions = bs.run((0, 1), k=3, depth=3)
        self.assertTrue(nodes == [(0, 1), (1, 2), (4, 3), (9, 14)])
