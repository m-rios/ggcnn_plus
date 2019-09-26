from unittest import TestCase
import keras
from core.network_optimization import NetworkOptimization
from core.network import Network
import numpy as np


class TestNetworkOptimization(TestCase):
    def test_expand(self):
        node = Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        op = NetworkOptimization(eval_method='iou', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5')
        children, scores, actions = op.expand(node)
        self.assertTrue(len(children) == 6)
        self.assertTrue((actions == ['deeper_conv_1', 'wider_conv_1', 'deeper_conv_2', 'wider_conv_2', 'deeper_conv_3', 'wider_conv_3']))

    def test_evaluate(self):
        node = Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        op = NetworkOptimization(eval_method='iou', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5')
        op.min_iou = 0.
        ev1 = op.evaluate(node)
        op.min_iou = 1.
        ev2 = op.evaluate(node)
        self.assertTrue(ev1 > 0)
        self.assertTrue(ev2 == 0)

    def test_run(self):
        node = Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        op = NetworkOptimization(eval_method='iou', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5', epochs=1, debug=True)
        [nodes, scores, actions] = op.run(node)
        print 'Nodes: {}'.format(nodes)
        print 'Scores: {}'.format(scores)
        print 'Actions: {}'.format(actions)

    def test_run_short(self):
        node = Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        op = NetworkOptimization(eval_method='iou', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5', epochs=0, debug=True)
        [nodes, scores, actions] = op.run(node)
        print 'Nodes: {}'.format(nodes)
        print 'Scores: {}'.format(scores)
        print 'Actions: {}'.format(actions)

    def test_run_short_transpose(self):
        node = Network(model_fn='../data/networks/shallow/epoch_50_model.hdf5')
        op = NetworkOptimization(eval_method='iou', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5', epochs=0, debug=True, expand_transpose=True)
        [nodes, scores, actions] = op.run(node, depth=2, k=1)
        print 'Nodes: {}'.format(nodes)
        print 'Scores: {}'.format(scores)
        print 'Actions: {}'.format(actions)

    def test_evaluate_loss(self):
        node = Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        op = NetworkOptimization(eval_method='loss', dataset_fn='../data/datasets/preprocessed/jacquard_samples.hdf5',
                                 epochs=1, debug=True)
        op.evaluate(node)
