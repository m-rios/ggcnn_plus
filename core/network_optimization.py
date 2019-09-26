import numpy as np
import h5py
import core.network as net
import os
import datetime

from utils.beam_search import BeamSearch
from utils.dataset import DatasetGenerator


class NetworkOptimization(BeamSearch):

    def __init__(self, eval_method, dataset_fn, min_iou=0.25, debug=False, epochs=10, batch_sz=4, results_path='.', retrain_epochs=0, expand_transpose=False):
        """
        Optimize GGCNN using Beam Search
        :param eval_method: string indicating method to use for evaluation. Either 'sim' or 'iou'
        :param dataset_fn: path to the hdf5 dataset used for the evaluation
        :param min_iou: minimum iou value to consider a grasp successful (used for iou evaluation only)
        :param debug: if set to True algorithm will be verbose
        :param epochs: number of epochs to retrain the expanded models for
        :param results_path: path where the results folder will written
        """
        assert eval_method == 'iou' or eval_method == 'sim' or eval_method == 'loss'

        self.eval_method = eval_method
        self.min_iou = min_iou
        self.dataset_fn = dataset_fn
        self._dataset = h5py.File(self.dataset_fn, 'r')
        self.train_generator = DatasetGenerator(dataset_fn, batch_sz, 'train')
        self.test_generator = DatasetGenerator(dataset_fn, batch_sz, 'test')
        self.epochs = epochs
        self.batch_sz = batch_sz
        self.retrain_epochs = retrain_epochs
        self.expand_transpose = expand_transpose

        self.current_depth = 0

        self.results_path = results_path
        self.log_fn = os.path.join(self.results_path, 'results.txt')

        if self.eval_method == 'iou':
            self.scenes = self._dataset['test']['img_id'][:]
            self.depth = self._dataset['test']['depth_inpainted'][:]
            self.bbs = self._dataset['test']['bounding_boxes'][:]
        elif self.eval_method == 'loss':
            self.x_test = np.expand_dims(np.array(self._dataset['test/depth_inpainted']), -1)
            point_test = np.expand_dims(np.array(self._dataset['test/grasp_points_img']), -1)
            angle_test = np.array(self._dataset['test/angle_img'])
            cos_test = np.expand_dims(np.cos(2 * angle_test), -1)
            sin_test = np.expand_dims(np.sin(2 * angle_test), -1)
            grasp_width_test = np.expand_dims(np.array(self._dataset['test/grasp_width']), -1)
            grasp_width_test = np.clip(grasp_width_test, 0, 150) / 150.0
            self.y_test = [point_test, cos_test, sin_test, grasp_width_test]
        else:
            raise NotImplemented('sim evaluation is still not supported')

        super(NetworkOptimization, self).__init__(debug=debug, log_fn=self.log_fn)
        self.log.info("""
        ARCHITECTURE OPTIMIZATION PARAMETERS
        ====================================\n\neval_method: {}\ndataset_fn: {}\nmin_iou: {}
        epochs: {}\nretrain_epochs: {}\nbatch_sz:{}\nexpand_transpose:{}\n\n""".format(eval_method, dataset_fn, min_iou, epochs,
                                                                  retrain_epochs, batch_sz, expand_transpose))

    def expand(self, node):
        children = []
        scores = []
        actions = []

        for layer_idx in node.get_expandable_layer_idxs(transpose=self.expand_transpose):
            is_conv = type(node.model.layers[layer_idx]).__name__ == 'Conv2D'
            prefix = '' if is_conv else 'de'
            deeper = node.deeper(layer_idx)
            wider = node.wider(layer_idx)
            self.log.info('Training {}'.format(deeper))
            deeper.train(self.train_generator, self.batch_sz, self.epochs, verbose=0)
            self.log.info('Training {}'.format(wider))
            wider.train(self.train_generator, self.batch_sz, self.epochs, verbose=0)
            children += [deeper, wider]
            scores += [self.evaluate(deeper), self.evaluate(wider)]
            actions += ['deeper_{}conv_{}'.format(prefix, layer_idx), 'wider_{}conv_{}'.format(prefix,layer_idx)]

        return children, scores, actions

    def _evaluate_iou(self, node):
        positions, angles, widths = node.predict(self.depth)
        succeeded, failed = net.calculate_iou_matches(positions, angles, self.bbs,
                                                      no_grasps=1,
                                                      grasp_width_out=widths,
                                                      min_iou=self.min_iou)
        return float(len(succeeded))/(len(succeeded) + len(failed))

    def _evaluate_loss(self, node):
        # node.compile(optimizer='rmsprop', loss='mean_squared_error')
        return np.mean(node.model.evaluate(self.x_test, self.y_test, verbose=0))

    def _evaluate_sim(self, node):
        raise NotImplemented('sim evaluation is still not supported')

    def evaluate(self, node):
        self.log.info("Evaluating network {}".format(str(node)))

        if self.eval_method == 'iou':
            result = self._evaluate_iou(node)
        elif self.eval_method == 'sim':
            result = self._evaluate_sim(node)
        elif self.eval_method == 'loss':
            result = self._evaluate_loss(node)
        else:  # Failed sanity check
            raise ValueError('Unrecognised value of eval_method. Expected either sim or iou. Received {}'.format(
                self.eval_method))

        self.log.info("Evaluated. Result: {}".format(result))

        return result

    def post_lookahead(self, node):
        node.train(self.train_generator, self.batch_sz, self.retrain_epochs, verbose=0)
        name = 'arch_{}_depth_{}_model.hdf5'.format(node, self.current_depth)
        node.model.save(os.path.join(self.results_path, name))
        self.current_depth += 1
        return node
