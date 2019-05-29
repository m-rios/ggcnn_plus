from unittest import TestCase
from core.network import Network
import numpy as np
import keras


class TestNetwork(TestCase):
    def test_deeper(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.deeper(layer=2)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)

        self.assertTrue(len(network.model.layers) == len(network2.model.layers) - 1)
        for o1, o2 in zip(output1, output2):
            self.assertTrue((o1 == o2).all())

    def test_wider(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.wider(layer=2)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)

        import pylab as plt
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(output1[0].squeeze())
        plt.subplot(1, 2, 2)
        plt.imshow(output2[0].squeeze())
        plt.show()

        self.assertTrue(len(network.model.layers) == len(network2.model.layers))
        for o1, o2 in zip(output1, output2):
            print 'o1: {}\no2: {}'.format(o1.flatten(), o2.flatten())
            self.assertTrue((np.round(o1, 2) == np.round(o2, 2)).all())

    def test_wider_2(self):
        input =  np.reshape(np.array([[7, 8, 9],
                                      [5, 6, 4],
                                      [4, 1, 3]]), (1, 3, 3, 1))
        n_filters = 2
        i = keras.layers.Input((3, 3, 1))
        l1 = keras.layers.Conv2D(n_filters, padding='same', kernel_size=(3,3), activation='relu')(i)
        l2 = keras.layers.Conv2D(n_filters*2, padding='same', kernel_size=(3,3), activation='relu')(l1)
        l3 = keras.layers.Conv2DTranspose(n_filters, padding='same', kernel_size=(3,3), activation='relu')(l2)

        model = keras.models.Model(i, [l3, l3])

        network = Network(model=model)
        network2 = network.wider(2)

        self.assertTrue((network.model.predict(input) == network2.model.predict(input))).all()


    def test_conv_layer_idxs(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        layers = network.conv_layer_idxs
        self.assertTrue([1, 2, 3] == layers)
