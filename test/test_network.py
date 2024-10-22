from unittest import TestCase
from core.network import Network
from utils.dataset import DatasetGenerator
import numpy as np
import keras


class TestNetwork(TestCase):
    def test_deeper(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.deeper(layer=2)
        network3 = network.deeper(layer=5)
        network4 = network.deeper(layer=6)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)
        output3 = network3.predict(input_img)
        output4 = network4.predict(input_img)

        self.assertTrue(str(network2) == 'C9x9x32_C5x5x16_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32')
        self.assertTrue(str(network3) == 'C9x9x32_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T5x5x16_T9x9x32')
        self.assertTrue(str(network4) == 'C9x9x32_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32_T9x9x32')
        for o in [output2, output3, output4]:
            self.assertTrue((o[0] == output1[0]).all())

    def test_wider(self):
        layer = 1
        layer2 = 4
        layer3 = 6
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        network2 = network.wider(layer=layer)
        network3 = network.wider(layer=layer2)
        network4 = network.wider(layer=layer3)
        input_img = np.expand_dims(np.load('depth_inpainted.npy'), axis=2)
        input_img = np.expand_dims(input_img, axis=0)
        output1 = network.predict(input_img)
        output2 = network2.predict(input_img)
        output3 = network3.predict(input_img)
        output4 = network4.predict(input_img)

        self.assertTrue(len(network2.model.layers) == len(network.model.layers))
        self.assertTrue(network2.model.layers[layer].filters == network.model.layers[layer].filters*2)
        self.assertTrue(network2.model.layers[layer + 1].output.shape == network2.model.layers[layer + 1].output.shape)

        import pylab as plt
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(output1[0].squeeze())
        plt.subplot(2, 2, 2)
        plt.imshow(output2[0].squeeze())
        plt.subplot(2, 2, 3)
        plt.imshow(output3[0].squeeze())
        plt.subplot(2, 2, 4)
        plt.imshow(output4[0].squeeze())
        plt.show()

        # for o1, o2 in zip(output1, output2):
        #     print 'o1: {}\no2: {}'.format(o1.flatten(), o2.flatten())
        #     self.assertTrue((np.round(o1, 2) == np.round(o2, 2)).all())

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

        mh1 = keras.models.Model(m1.layers[0].output, hidden1)
        mh2 = keras.models.Model(m2.layers[0].output, hidden2)

        o1 = mh1.predict(input_img)
        o2 = mh2.predict(input_img)

        for i in range(o1.shape[3]):
            print i, np.amax(np.abs(o1[:,:,:,i] - o2[:,:,:,i]))

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

        model2 = network2.copy_model()

        # Hidden outputs
        modelh1 = keras.models.Model(model.layers[0].output, model.layers[2].output)
        modelh2 = keras.models.Model(model2.layers[0].output, model2.layers[2].output)

        hidden1 = modelh1.predict(input)
        hidden2 = modelh2.predict(input)

        self.assertTrue((np.round(hidden1, 3) == np.round(hidden2, 3)).all())

        final1 = model.predict(input)
        final2 = model2.predict(input)

        self.assertTrue((np.round(final1, 3) == np.round(final2, 3)).all())

    def test_conv_layer_idxs(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        layers = network.conv_layer_idxs
        self.assertTrue([1, 2, 3] == layers)
        deeper = network.deeper(3)
        layers2 = deeper.conv_layer_idxs
        self.assertTrue(layers2 == [1, 2, 3, 4])

    def test_get_expandable_layer_idxs(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        layers = network.get_expandable_layer_idxs()
        self.assertTrue([1, 2, 3] == layers)
        deeper = network.deeper(3)
        layers2 = deeper.conv_layer_idxs
        self.assertTrue(layers2 == [1, 2, 3, 4])
        layers = network.get_expandable_layer_idxs(transpose=True)
        self.assertTrue([1, 2, 3, 4, 5, 6] == layers)

    def test_train(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        train_generator = DatasetGenerator('../data/datasets/preprocessed/jacquard_samples.hdf5', 4)
        network.train(train_generator, 4, 2)
        network.train(train_generator, 4, 0)

    def test_str(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        self.assertTrue(str(network) == 'C9x9x32_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32')

    def test_get_connectivity(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        connectivity = network.get_connectivity()
        connectivity2 = network.get_connectivity(layer_idx=3)
        self.assertTrue(connectivity == [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
        self.assertTrue(connectivity2 == [-1, 0, 1, 2, 3, 3, 3, 3])

    def test_reconnect_model(self):
        network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        layer = keras.layers.Conv2D(8, kernel_size=(3,3),
                                                 strides=(2,2),
                                                 padding='same',
                                                 activation='relu',
                                                 name='test_layer')
        layer2 = keras.layers.Conv2D(8, kernel_size=(3,3),
                                    strides=(2,2),
                                    padding='same',
                                    activation='relu',
                                    name='test_layer')
        layer3 = keras.layers.Conv2D(32, kernel_size=(3,3),
                                    strides=(2,2),
                                    padding='same',
                                    activation='relu',
                                    name='test_layer')
        network1 = Network(model=network.reconnect_model(3, [layer]))
        self.assertTrue(str(network1) == 'C9x9x32_C5x5x16_C3x3x8_C3x3x8_T3x3x8_T5x5x16_T9x9x32')
        network2 = Network(model=network.reconnect_model(3, [layer2], replace=True))
        self.assertTrue(str(network2) == 'C9x9x32_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32')
        network3 = Network(model=network.reconnect_model(6, [layer3]))
        self.assertTrue(str(network3) == 'C9x9x32_C5x5x16_C3x3x8_T3x3x8_T5x5x16_T9x9x32_C3x3x32')

    def test_subsequent_expansions(self):
        network = Network(model_fn='../data/networks/shallow/epoch_50_model.hdf5')
        network2 = network.wider(2)
        network3 = network2.wider(2)
        self.assertTrue(True)
