from unittest import TestCase
from core.network import Network
import numpy as np


class TestNetwork(TestCase):
    def test_deeper(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.deeper(layer=2)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)

        for o1, o2 in zip(output1, output2):
            self.assertTrue((o1 == o2).all())

    def test_wider(self):
        self.fail()
