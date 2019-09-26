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

import numpy as np

from unittest import TestCase
from utils.beam_search2 import BeamSearch


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

        def expand(self, node):
            children = []
            actions = []

            for action_idx, child in enumerate(self.children[node[0]]):
                children.append(child)
                actions.append('child {}'.format(action_idx))

            return children, actions

        def evaluate(self, node):
            return node[1]

    def test_lookahead(self):
        bs = self.BS()
        nodes, parents, scores, actions, bw = bs.lookahead((0, 1), k=3, depth=2)
        self.assertTrue(nodes == [(0, 1), (2, 4), (1, 2), (7, 9), (5, 7), (3, 6)])
        self.assertTrue(parents == [-1, 0, 0, 1, 2, 2])
        self.assertTrue(scores == [-1, 4, 2, 9, 7, 6])

        nodes, parents, scores, actions, bw = bs.lookahead((0, 1), k=3, depth=3)
        self.assertTrue(nodes == [(0, 1), (2, 4), (1, 2), (7, 9), (5, 7), (3, 6), (10, 10)])
        self.assertTrue(parents == [-1, 0, 0, 1, 2, 2, 4])
        self.assertTrue(scores == [-1, 4, 2, 9, 7, 6, 10])

    def test_run(self):
        bs = self.BS()
        nodes, scores, actions = bs.run((0, 1), k=3, depth=3)
        self.assertTrue(nodes == [(0, 1), (1, 2), (4, 3), (9, 14)])
        self.assertTrue(scores == [1, 2, 3, 14])
        self.assertTrue(actions == [None, 'child 0', 'child 1', 'child 0'])

    def test_find_parent_idx(self):
        parents_idx = [-1, 0, 0, 1, 1, 1, 2, 2, 2, 4, 5, 6]
        self.assertTrue(BeamSearch.find_parent_idx(11, parents_idx) == 2)
        self.assertTrue(BeamSearch.find_parent_idx(9, parents_idx) == 1)

    # def test_lookahead_only(self):
    #     bs = self.BS()
    #     nodes, parents, scores, actions, bw = bs.lookahead((0, 1), k=6, depth=3)
    #
    #     beam_scores = scores[-bw:]
    #     best_leaf_idx = (len(scores) - bw + np.argsort(beam_scores)[-1])
    #     best_path = [best_leaf_idx]
    #
    #     while parents[best_path[0]] != 0:
    #         best_path.insert(0, parents[best_path[0]])
    #
    #     print best_path
    #     self.assertTrue([1, 4, 9] == best_path)
