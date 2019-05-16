from utils.beam_search import BeamSearch


class NetworkOptimization(BeamSearch):

    def __init__(self, debug=False):
        super(NetworkOptimization, self).__init__(debug=debug)

    def expand(self, node):
        pass

    def evaluate(self, node):
        pass
