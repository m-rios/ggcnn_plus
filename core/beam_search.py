from abc import ABCMeta, abstractmethod
import numpy as np
import logging


class BeamSearch:
    __metaclass__ = ABCMeta

    def __init__(self, debug=False):
        import logging
        self.log = logging

        if debug:
            self.log.basicConfig(level=self.log.DEBUG)
        else:
            self.log.basicConfig(level=self.log.INFO)

        super(BeamSearch, self).__init__()

    @abstractmethod
    def node_actions(self, node):
        """
        Generates a list of actions that branch from the given node
        :param node: tree node from which the actions branch
        :return: A list of tuples of 2 elements. tuple[0] is a function that will execute the action. tuple[1] contains
        parameters that might be needed for tuple[0]
        """
        raise NotImplemented

    @abstractmethod
    def evaluate(self, node):
        """
        Heuristic that determines how good a node is
        :param node: node to be evaluated
        :return: a score that evaluates the node
        """
        raise NotImplemented

    def lookahead(self, root, k, depth):
        """
        Explores nodes ahead of root to determine which immediate root's child is best to expand next
        :param root: node from which lookahead is performed
        :param k: beam width
        :param depth: depth limit of the lookahead tree
        :return: [nodes, parents_idx, scores, actions, beam_width]; nodes: list of nodes that have been explored;
        parents_idx: list of ids referencing nodes as parent. I.e. parents_idx[c] is the index in nodes of the parent of
        node c; scores: list of the scores of the nodes; actions: list of the actions that resulted in each node;
        beam_width: last beam width used (in case that available nodes at leaf depth is lower than k)
        """
        # Initialize return values for root
        nodes = [root]
        parents_idx = [-1]
        scores = [-1]
        actions = [None]
        beam_width = None

        queue = [root]  # Nodes to be expanded

        for d in range(depth):
            children = []
            children_parents = []
            children_scores = []
            children_actions = []

            for node_idx, node in enumerate(queue):
                children_actions += self.node_actions(node)
                for action, args in children_actions:
                    child = action(node, args)
                    children.append(child)
                    children_parents.append(node_idx)
                    children_scores.append(self.evaluate(child))

            beam_width = min(len(children), k)
            sort_idx = np.argsort(children_scores)[::-1]  # Sort indices descending
            selected_idx = sort_idx[:beam_width]  # Trim selection by beam_width

            nodes.append(children[selected_idx])
            parents_idx.append(children_parents[selected_idx])
            scores.append(children_scores[selected_idx])
            actions.append(children_actions[selected_idx])

            queue = children[selected_idx]

        return nodes, parents_idx, scores, actions, beam_width

    def post_lookahead(self, node):
        """
        Facilitates any possible postprocessing on the best node after lookahead.
        By default does nothing. Override to customize
        :param node: node to postprocess
        :return: postprocessed node
        """
        return node

    def run(self, node, k=3, depth=2):
        """
        Performs Breadth-First-Search with lookahead but limiting the expanding nodes by a beam width (k)
        :param node: Starting node of the search
        :param k: Beam width, max number of nodes that will be expanded at a given depth
        :param depth: Max depth the search will explore
        :return: [node, score, node_actions]; node: the node with the highest evaluation after the beam search; score:
        the score of the node; node_actions: a list of the actions that took to reach node
        """
        root = node

        for d in range(1, depth+1)[::-1]:  # Decrement d on each step so that max explored level is depth
            nodes, parents_idx, scores, actions, beam_width = self.lookahead(root, k, depth)
            beam_score = scores[-beam_width:]
            best_node_idx = np.argsort(beam_score)[::-1]
            root = self.post_lookahead(nodes[best_node_idx])
