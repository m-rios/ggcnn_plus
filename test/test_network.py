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
        layer = 1
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.wider(layer=layer)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)

        self.assertTrue(len(network2.model.layers) == len(network.model.layers))
        self.assertTrue(network2.model.layers[layer].filters == network.model.layers[layer].filters*2)
        self.assertTrue(network2.model.layers[layer + 1].output.shape == network2.model.layers[layer + 1].output.shape)

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

    def test_wider_at_hidden(self):
        layer = 2
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.wider(layer=layer)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = np.ones(input_img.shape)

        m1 = network.model
        m2 = network2.model
        hidden1 = m1.layers[layer + 1](m1.layers[layer].output)
        hidden2 = m2.layers[layer + 1](m2.layers[layer].output)

        mh1 = keras.models.Model(m1.input, hidden1)
        mh2 = keras.models.Model(m2.input, hidden2)

        o1 = mh1.predict(input_img)
        o2 = mh2.predict(input_img)

        for i in range(o1.shape[3]):
            print i, np.amax(np.abs(o1[:,:,:,i] - o2[:,:,:,i]))

        pass

    def test_wider_on_smaller_network(self):
        input =  np.reshape(np.array([[7, 8, 9],
                                      [5, 6, 4],
                                      [4, 1, 3]]), (1, 3, 3, 1)).astype(np.float64)
        n_filters = 2
        i = keras.layers.Input((3, 3, 1))
        l1 = keras.layers.Conv2D(n_filters, padding='same', kernel_size=(3,3), activation='relu')(i)
        l2 = keras.layers.Conv2D(n_filters*2, padding='same', kernel_size=(3,3), activation='relu')(l1)
        l3 = keras.layers.Conv2DTranspose(n_filters, padding='same', kernel_size=(3,3), activation='relu')(l2)

        model = keras.models.Model(i, l3)

        network = Network(model=model)
        network2 = network.wider(1)

        # Hidden outputs
        modelh1 = keras.models.Model(i, model.layers[2](model.layers[1].output))
        modelh2 = keras.models.Model(i, network2.model.layers[2](network2.model.layers[1].output))

        hidden1 = modelh1.predict(input)
        hidden2 = modelh2.predict(input)
        print hidden1
        print hidden2
        # print np.moveaxis(hidden2, [0, 1, 2, 3], [2, 3, 0, 1])
        print (hidden1 == hidden2).all()

        self.assertTrue((network.model.predict(input) == network2.model.predict(input)).all())

    def test_conv_layer_idxs(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        layers = network.conv_layer_idxs
        self.assertTrue([1, 2, 3] == layers)
