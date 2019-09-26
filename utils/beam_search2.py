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
        :return: children, actions; children: a list with the children of node;  actions: a list of the actions that
        generated each child
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

    @staticmethod
    def find_parent_idx(leaf_idx, parents_idx):
        """
        Finds the topmost parent of a node by its idx in the trace
        :param leaf_idx: Index of leaf in the trace
        :param parents_idx: Index of the parents in the trace for the correspondent trace idx
        :return: Index in the trace of the parent for leaf_idx
        """
        parent_idx = parents_idx[leaf_idx]
        while parents_idx[parent_idx] != 0:
            parent_idx = parents_idx[parent_idx]

        return parent_idx

    def lookahead(self, roots, depth):
        """
        Explores nodes ahead of root to determine which immediate root's child is best to expand next
        :param roots: node from which lookahead is performed
        :param depth: depth limit of the lookahead tree
        :return: [nodes, parents_idx, scores, actions, beam_width]; nodes: list of nodes that have been explored;
        parents_idx: list of ids referencing nodes as parent. I.e. parents_idx[c] is the index in nodes of the parent of
        node c; scores: list of the scores of the nodes; actions: list of the actions that resulted in each node;
        beam_width: last beam width used (in case that available nodes at leaf depth is lower than k)
        """
        self.log.debug('depth = {}'.format(depth))
        # Initialize return values for root
        nodes = roots
        parents_idx = [-1]
        actions = [None]

        queue = roots  # Nodes to be expanded
        parent_idx = 0

        for d in range(depth):
            self.log.debug('d = {}'.format(d))
            children = []
            children_parents = []
            children_actions = []

            for node in queue:
                node_children, node_scores, node_actions = self.expand(node)
                children += node_children
                children_parents += [parent_idx]*len(node_children)
                children_actions += node_actions
                parent_idx += 1

            self.log.debug('queue = {}'.format(queue))
            self.log.debug('children = {}'.format(children))
            self.log.debug('children_parents = {}'.format(children_parents))
            self.log.debug('children_actions = {}'.format(children_actions))

            queue = children

            nodes += queue
            parents_idx += children_parents
            actions += children_actions

        return nodes, parents_idx, actions, queue

    def backtrack(self, leafs, parents_idx, k):
        nodes, parents_idx, actions, leaf_nodes = self.lookahead(roots, k, d)
        self.log.info('Best child at depth {} found'.format(depth - d))

        self.log.debug('nodes = {}'.format(nodes))
        self.log.debug('parents_idx = {}'.format(parents_idx))

        # Compute leaf_node scores to find beam
        beam_scores = []
        for leaf in leaf_nodes:
            beam_scores += self.evaluate(leaf)

        beam_idxs = np.argsort(beam_scores)[-1:-(k + 1):-1]

        self.log.debug('beam_idx = {}'.format(beam_idxs))

        # Find best nodes at immediate next depth
        best_parents_idx = set()
        for beam_idx in beam_idxs:
            best_parents_idx.add(self.find_parent_idx(beam_idx, parents_idx))

        best_parents_idx = list(best_parents_idx)
        best_parents = nodes[best_parents_idx]
        self.log.debug('best_parents = {}'.format(best_parents))

        roots = []
        for best_parent in best_parents:
            roots.append(self.post_lookahead(best_parent))

        nodes_trace += roots
        scores_trace += [self.evaluate(root) for root in roots]
        actions_trace += [actions[best_parent_idx] for best_parent_idx in best_parents_idx]

        return beam_idx, selected_idx

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
        roots = [node]

        nodes, parents_idx, actions, leaf_nodes = self.lookahead(roots, depth)
        beam_idx, selected_leaves_idx = self.backtrack(leaf_nodes, parents_idx, k)

        roots = leaf_nodes[selected_leaves_idx]

        for d in range(2, depth+1):
            self.log.info('Starting lookahead at depth {}'.format(d))
            nodes, parents_idx, actions, leaf_nodes = self.lookahead(roots, 1)

            # Next nodes to expand
            beam_idx, selected_leaves_idx = self.backtrack(leaf_nodes, parents_idx, k)
            roots = leaf_nodes[selected_leaves_idx]

        beam_nodes = nodes[beam_idx]
        scores = [self.evaluate(node) for node in beam_nodes]
        best_node_idx = np.argmax(scores)

        return beam_nodes
        return nodes_trace, scores_trace, actions_trace


