from abc import ABCMeta, abstractmethod
import numpy as np


class BeamSearch:
    __metaclass__ = ABCMeta

    def __init__(self, debug=False, log_fn=None):
        import logging
        self.log = logging
        level = self.log.DEBUG if debug else self.log.INFO
        fmt = "[%(levelname)s]: [%(asctime)s]: %(funcName)s():%(lineno)i: %(message)s"
        self.log.basicConfig(level=level, format=fmt)
        if log_fn is not None:
            self.log.getLogger().addHandler(logging.FileHandler(filename=log_fn))

        super(BeamSearch, self).__init__()

    @abstractmethod
    def expand(self, node):
        """
        Expands the input node into its children
        :param node: node to be expanded
        :return: children, scores, actions; children: a list with the children of node; scores: a list of the score of
        each child; actions: a list of the actions that generated each child
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
        self.log.debug('k = {}; depth = {}'.format(k, depth))
        # Initialize return values for root
        nodes = [root]
        parents_idx = [-1]
        scores = [-1]
        actions = [None]
        beam_width = None

        queue = [root]  # Nodes to be expanded
        parent_idx = 0

        for d in range(depth):
            self.log.debug('d = {}'.format(d))
            children = []
            children_parents = []
            children_scores = []
            children_actions = []

            for node in queue:
                node_children, node_scores, node_actions = self.expand(node)
                children += node_children
                children_parents += [parent_idx]*len(node_children)
                children_scores += node_scores
                children_actions += node_actions
                parent_idx += 1

            beam_width = min(len(children), k)
            sort_idx = np.argsort(children_scores)[::-1]  # Sort indices descending
            selected_idx = sort_idx[:beam_width]  # Trim selection by beam_width

            self.log.debug('queue = {}'.format(queue))
            self.log.debug('children = {}'.format(children))
            self.log.debug('children_parents = {}'.format(children_parents))
            self.log.debug('children_scores = {}'.format(children_scores))
            self.log.debug('children_actions = {}'.format(children_actions))
            self.log.debug('selected_idx = {}'.format(selected_idx))

            queue = [children[s] for s in selected_idx]

            nodes += queue
            parents_idx += [children_parents[s] for s in selected_idx]
            scores += [children_scores[s] for s in selected_idx]
            actions += [children_actions[s] for s in selected_idx]

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
        :return: [nodes_trace, scores_trace, actions_trace]; node_trace: a list with the nodes that resulted from each
        iteration; scores_trace: a list with the scores of each node; actions_trace: a list with the actions that
        produced each node. The lists are ordered from top to bottom in the tree hierarchy. To get the last node
        expanded (should be the best) use nodes_trace[-1]
        """
        root = node
        nodes_trace = [node]
        scores_trace = [self.evaluate(node)]
        actions_trace = [None]

        for d in range(1, depth+1)[::-1]:  # Decrement d on each step so that max explored level is depth
            self.log.info('Starting lookahead at depth {}'.format(depth-d))
            nodes, parents_idx, scores, actions, beam_width = self.lookahead(root, k, d)
            self.log.info('Best child at depth {} found'.format(depth-d))

            self.log.debug('nodes = {}'.format(nodes))
            self.log.debug('parents_idx = {}'.format(parents_idx))

            beam_scores = scores[-beam_width:]
            best_leaf_idx = (len(scores) - beam_width + np.argsort(beam_scores)[-1])

            self.log.debug('best_leaf_idx = {}'.format(best_leaf_idx))

            best_parent_idx = parents_idx[best_leaf_idx]
            while parents_idx[best_parent_idx] != 0:
                best_parent_idx = parents_idx[best_parent_idx]

            best_parent = nodes[best_parent_idx]
            self.log.debug('best_parent = {}'.format(best_parent))

            root = self.post_lookahead(best_parent)

            nodes_trace.append(root)
            scores_trace.append(self.evaluate(root))
            actions_trace.append(actions[best_parent_idx])

        return nodes_trace, scores_trace, actions_trace


