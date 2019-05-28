import h5py
import core.network as net

from utils.beam_search import BeamSearch


class NetworkOptimization(BeamSearch):

    def __init__(self, eval_method, dataset_fn, min_iou=0.25, debug=False):
        """
        Optimize GGCNN using Beam Search
        :param eval_method: string indicating method to use for evaluation. Either 'sim' or 'iou'
        :param dataset_fn: path to the hdf5 dataset used for the evaluation
        :param min_iou: minimum iou value to consider a grasp successful (used for iou evaluation only)
        :param debug: if set to True algorithm will be verbose
        """
        assert eval_method == 'iou' or eval_method == 'sim'

        self.eval_method = eval_method
        self.min_iou = min_iou
        self.dataset_fn = dataset_fn
        self._dataset = h5py.File(self.dataset_fn, 'r')

        if self.eval_method == 'iou':
            self.scenes = self._dataset['test']['img_id'][:]
            self.depth = self._dataset['test']['depth_inpainted'][:]
            self.bbs = self._dataset['test']['bounding_boxes'][:]
        else:
            pass

        super(NetworkOptimization, self).__init__(debug=debug)

    def expand(self, node):
        pass

    def _evaluate_iou(self, node):
        positions, angles, widths = node.predict(self.depth)
        succeeded, failed = net.calculate_iou_matches(positions, angles, self.bbs,
                                                      no_grasps=1,
                                                      grasp_width_out=widths,
                                                      min_iou=self.min_iou)
        return float(len(succeeded))/(len(succeeded) + len(failed))

    def _evaluate_sim(self, node):
        raise NotImplemented('sim evaluation is still not supported')

    def evaluate(self, node):
        if self.eval_method == 'iou':
            return self._evaluate_iou(node)
        elif self.eval_method == 'sim':
            return self._evaluate_sim(node)
        else:  # Failed sanity check
            raise ValueError('Unrecognised value of eval_method. Expected either sim or iou. Received {}'.format(self.eval_method))
