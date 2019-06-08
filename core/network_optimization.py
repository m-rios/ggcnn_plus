import h5py
import core.network as net
import os
import datetime

from utils.beam_search import BeamSearch
from utils.dataset import DatasetGenerator


class NetworkOptimization(BeamSearch):

    def __init__(self, eval_method, dataset_fn, min_iou=0.25, debug=False, epochs=10, batch_sz=4, results_path='.'):
        """
        Optimize GGCNN using Beam Search
        :param eval_method: string indicating method to use for evaluation. Either 'sim' or 'iou'
        :param dataset_fn: path to the hdf5 dataset used for the evaluation
        :param min_iou: minimum iou value to consider a grasp successful (used for iou evaluation only)
        :param debug: if set to True algorithm will be verbose
        :param epochs: number of epochs to retrain the expanded models for
        :param results_path: path where the results folder will written
        """
        assert eval_method == 'iou' or eval_method == 'sim'

        self.eval_method = eval_method
        self.min_iou = min_iou
        self.dataset_fn = dataset_fn
        self._dataset = h5py.File(self.dataset_fn, 'r')
        self.train_generator = DatasetGenerator(dataset_fn, batch_sz, 'train')
        self.test_generator = DatasetGenerator(dataset_fn, batch_sz, 'test')
        self.epochs = epochs
        self.batch_sz = batch_sz

        self.current_depth = 0

        self.results_path = results_path
        self.log_fn = os.path.join(self.results_path, 'results.txt')

        if self.eval_method == 'iou':
            self.scenes = self._dataset['test']['img_id'][:]
            self.depth = self._dataset['test']['depth_inpainted'][:]
            self.bbs = self._dataset['test']['bounding_boxes'][:]
        else:
            raise NotImplemented('sim evaluation is still not supported')

        super(NetworkOptimization, self).__init__(debug=debug, log_fn=self.log_fn)
        self.log.info("""
        ARCHITECTURE OPTIMIZATION PARAMETERS
        ====================================\n\neval_method: {}\ndataset_fn: {}\nmin_iou: {}
        epochs: {}\nbatch_sz={}\n\n""".format(eval_method, dataset_fn, min_iou, epochs, batch_sz))

    def expand(self, node):
        children = []
        scores = []
        actions = []

        for layer_idx in node.conv_layer_idxs:
            deeper = node.deeper(layer_idx)
            wider = node.wider(layer_idx)
            self.log.info('Training {}'.format(deeper))
            deeper.train(self.train_generator, self.batch_sz, self.epochs, verbose=0)
            self.log.info('Training {}'.format(wider))
            wider.train(self.train_generator, self.batch_sz, self.epochs, verbose=0)
            children += [deeper, wider]
            scores += [self.evaluate(deeper), self.evaluate(wider)]
            actions += ['deeper_conv_{}'.format(layer_idx), 'wider_conv_{}'.format(layer_idx)]

        return children, scores, actions

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
        self.log.info("Evaluating network {}".format(str(node)))

        if self.eval_method == 'iou':
            result = self._evaluate_iou(node)
        elif self.eval_method == 'sim':
            result = self._evaluate_sim(node)
        else:  # Failed sanity check
            raise ValueError('Unrecognised value of eval_method. Expected either sim or iou. Received {}'.format(
                self.eval_method))

        self.log.info("Evaluated. Result: {}".format(result))

        return result

    def post_lookahead(self, node):
        name = 'arch_{}_depth_{}_model.hdf5'.format(node, self.current_depth)
        node.model.save(os.path.join(self.results_path, name))
        self.current_depth += 1
        return node
