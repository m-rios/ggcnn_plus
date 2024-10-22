import keras
from ggcnn.dataset_processing.grasp import detect_grasps
from ggcnn.dataset_processing.grasp import BoundingBoxes, BoundingBox
from skimage.filters import gaussian
from skimage.transform import rescale, resize
import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_iou_matches(grasp_positions_out, grasp_angles_out, ground_truth_bbs, no_grasps=1, grasp_width_out=None, min_iou=0.25):
    """
    Calculate a success score using the (by default) 25% IOU metric.
    Note that these results don't really reflect real-world performance.
    """
    succeeded = []
    failed = []
    for i in range(grasp_positions_out.shape[0]):
        grasp_position = grasp_positions_out[i, ].squeeze()
        grasp_angle = grasp_angles_out[i, :, :].squeeze()

        grasp_position = gaussian(grasp_position, 5.0, preserve_range=True)

        if grasp_width_out is not None:
            grasp_width = grasp_width_out[i, ].squeeze()
            grasp_width = gaussian(grasp_width, 1.0, preserve_range=True)
        else:
            grasp_width = None

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs[i, ].squeeze())
        gs = detect_grasps(grasp_position, grasp_angle, width_img=grasp_width, no_grasps=no_grasps, ang_threshold=0)
        for g in gs:
            if g.max_iou(gt_bbs) > min_iou:
                succeeded.append(i)
                break
        else:
            failed.append(i)

    return succeeded, failed


def get_output_plot(depth, position, angle, width, no_grasps=1, ground_truth=None):
    depth = depth.squeeze()
    angle = angle.squeeze()
    position = gaussian(position.squeeze(), 5.0, preserve_range=True)
    width = gaussian(width.squeeze(), 1.0, preserve_range=True)

    gs = detect_grasps(position, angle, width_img=width, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('Depth')
    ax.imshow(depth)

    if ground_truth is not None:
        for gt in ground_truth:
            g = BoundingBox(gt).as_grasp
            g.plot(ax, 'k')

    for g in gs:
        g.plot(ax, 'r')

    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('Grasp quality')
    ax.imshow(position, cmap='Reds', vmin=0, vmax=1)

    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Grasp angle')
    plot = ax.imshow(angle, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    plt.colorbar(plot)
    return fig


def plot_grasps(depth, gs):

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.set_axis_off()
    ax.imshow(depth, aspect='equal')
    ax.set_xlim([0, 299])
    ax.set_ylim([0, 299])
    colors = ['r', 'w', 'k', 'm', 'g']
    for g_idx, g in enumerate(gs):
        g.plot(ax, colors[g_idx])
        ax.text(g.center[1], g.center[0], str(g_idx + 1), color=colors[g_idx])

    plt.margins(0)

    return fig

def plot_output(depth, positions, angles, widths, no_grasps=1, ground_truth=None):
    fig = get_output_plot(depth, positions, angles, widths, no_grasps, ground_truth)
    plt.show(fig)
    plt.close(fig)


def save_output_plot(depth, positions, angles, widths, filename, no_grasps=1,
        ground_truth=None):
    fig = get_output_plot(depth, positions, angles, widths, no_grasps, ground_truth)
    plt.savefig(filename)
    plt.close(fig)


def get_grasps_from_output(position, angle, width, n_grasps=1):
    """
    Computes n_grasps given the an individual network input

    Args
    ----
    position: np.ndarray, shape(height, width, _)
    angle: np.ndarray, shape(height, width, _)
    width: np.ndarray, shape(height, width, _)
    n_grasps: number of grasps to extract from the image

    Returns
    -------
    A list of size n_grasps with Grasp objects
    """
    position = gaussian(position.squeeze(), 5.0, preserve_range=True)
    angle = angle.squeeze()
    width = gaussian(width.squeeze(), 1.0, preserve_range=True)
    gs = detect_grasps(position,
            angle,
            width_img=width,
            no_grasps=n_grasps)
    return gs


def subsample(image, factor=0.5):
    image_subsampled = rescale(image, factor, anti_aliasing=True, multichannel=False)
    return resize(image_subsampled, image.shape, anti_aliasing=True)


class Network:

    def __init__(self, model_fn=None, model=None):
        self.epoch = None

        if model is not None:
            self.model = model
        elif model_fn is not None:
            self.model = keras.models.load_model(model_fn)
            try:
                self.epoch = int(model_fn.split('_')[-2])
            except ValueError:
                print('Unable to extract epoch from model filename. Falling back to default value of None')
        else:
            raise ValueError('Either model_fn or model must be provided')

    def __str__(self):
        fields = []
        for layer_idx in self.get_expandable_layer_idxs(transpose=True):
            layer = self.model.layers[layer_idx]
            prefix = 'C' if type(layer).__name__ == 'Conv2D' else 'T'
            fields.append(prefix + 'x'.join(map(str, layer.kernel_size + (layer.filters,))))
        return '_'.join(fields)

    @property
    def height(self):
        return self.model.input.shape[1]

    @property
    def width(self):
        return self.model.input.shape[2]

    @property
    def conv_layer_idxs(self):
        """
        The indices of the hidden convolutional layers in the model
        """
        conv_layers = filter(lambda x: type(x[1]).__name__ == 'Conv2D', enumerate(self.model.layers))
        conv_layers = conv_layers[:-len(self.model.output)]
        return list(zip(*conv_layers)[0])

    def get_expandable_layer_idxs(self, transpose=False):
        valid_types = ['Conv2D']
        if transpose:
            valid_types.append('Conv2DTranspose')
        conv_layers = filter(lambda x: type(x[1]).__name__ in valid_types, enumerate(self.model.layers))
        conv_layers = conv_layers[:-len(self.model.output)]
        return list(zip(*conv_layers)[0])

    def copy_model(self):
        new_model = keras.models.clone_model(self.model)
        new_model.set_weights(self.model.get_weights())
        return new_model

    def reconnect_model_original(self, layer_idx, layers, replace=False):
        """
        Reconnects last_layer with the reminder of layers after layer_idx
        :param layer_idx: Index in the model of the first layer to be reconnected
        :param layers: An ordered list of the layers to be reconnected. Layers after the first one are connected to the first one
        :param replace: If True layers will replace existing layers at and bellow layer_idx. If False layers will be added
        bellow layer_idx
        :return: A copy of the reconnected model
        """
        if replace:
            start_layer_idx = layer_idx - 1
            end_layer_idx = layer_idx + len(layers)
        else:
            start_layer_idx = layer_idx
            end_layer_idx = layer_idx + len(layers)

        temp_model = self.copy_model()

        # Connect new layers
        input_layer = temp_model.layers[start_layer_idx]
        x = input_layer.output
        for layer in layers:
            x = layer(x)

        # Reconnect intermediate layers
        for hidden_layer in temp_model.layers[end_layer_idx:-len(temp_model.outputs)]:
            x = hidden_layer(x)
        # Reconnect output layers
        outputs = []
        for output_layer in temp_model.layers[-len(temp_model.outputs):]:
            outputs.append(output_layer(x))

        return keras.models.Model(temp_model.input, outputs)

    def get_connectivity_original(self, layer_idx=0, model=None):
        """
        Returns the connectivity of the model
        :param layer_idx: Layer from which to start retrieving connectivity
        :param model: model to get connectivity from (if None gets it from self)
        :return: A list where each element points to the layer with which that element as an inbound connection with
        """
        model = self.model if model is None else model
        assert len(model.layers) > 1, 'Model must have at least 2 layers for this function to be used. Current model only has {}'.format(len(model.layers))
        connectivity = [-1]
        indexed_layers = zip(range(len(model.layers[layer_idx:])), model.layers[layer_idx:])
        for layer in model.layers[layer_idx+1:]:
            inbound = layer._inbound_nodes[0].inbound_layers[0]
            c, _ = filter(lambda il: il[1] is inbound, indexed_layers)[0]
            connectivity.append(c)
        return connectivity

    def get_connectivity(self, layer_idx=0, model=None):
        # model = self.model if model is None else model
        model = self.model
        connectivity = [-1] + range(len(model.layers) - len(model.output) - 1 - layer_idx)
        connectivity += [connectivity[-1] + 1] * 4
        return connectivity

    def reconnect_model(self, layer_idx, layers, connectivity=None, replace=False):
        """
        Reconnects last_layer with the reminder of layers after layer_idx
        :param layer_idx: Index in the model of the first layer to be reconnected
        :param layers: An ordered list of the layers to be reconnected. Layers after the first one are connected to the first one
        :param connectivity: How each layer in layers is connected. If None layers are connected one after the other. An example
        to connect 2 deeper layers to a shallower one would be connectivity=[-1, 0, 0]
        :param replace: If True layers will replace existing layers at and bellow layer_idx. If False layers will be added
        bellow layer_idx
        :return: A copy of the reconnected model
        """
        if replace:
            start_layer_idx = layer_idx - 1
            end_layer_idx = layer_idx + len(layers)
        else:
            start_layer_idx = layer_idx
            end_layer_idx = layer_idx + len(layers)

        temp_model = self.copy_model()

        # Get new model connectivity
        submodel_layers = layers + temp_model.layers[end_layer_idx:]
        submodel_outputs = [temp_model.layers[start_layer_idx].output]
        if connectivity is None:
            connectivity = range(len(layers))
        else:
            connectivity = np.add(connectivity, 1).tolist()

        connectivity += np.add(self.get_connectivity(layer_idx=end_layer_idx-1, model=temp_model), len(layers))[1:].tolist()

        # Reconnect model
        for l_idx, layer in enumerate(submodel_layers):
            submodel_outputs.append(layer(submodel_outputs[connectivity[l_idx]]))

        return keras.models.Model(temp_model.input, submodel_outputs[-4:])

    def wider_orig(self, layer, factor=2):
        """
        Adds convolutional filters to a layer and adjusts weights using transfer learning
        :param layer: layer number that will be modified
        :param factor: value by which number of filters is increased
        :return: A copy of self with the updated layer
        """
        old_layer = self.model.layers[layer]
        old_layer_type = type(old_layer).__name__
        old_conv = old_layer_type == 'Conv2D'
        next_layer = self.model.layers[layer + 1]
        next_layer_type = type(next_layer).__name__
        next_conv = next_layer_type == 'Conv2D'
        n_filters = old_layer.filters
        name = old_layer.name + '_wider'

        assert old_layer_type == 'Conv2D' or old_layer_type == 'Conv2DTranspose'
        assert next_layer_type == 'Conv2D' or next_layer_type == 'Conv2DTranspose'

        # Replace old layer with a wider version and reconnect the graph
        if old_conv:
            modified_layer = keras.layers.Conv2D(n_filters*factor, kernel_size=old_layer.kernel_size,
                                                 strides=old_layer.strides,
                                                 padding='same',
                                                 activation=old_layer.activation,
                                                 name=name)
        else:
            modified_layer = keras.layers.Conv2DTranspose(n_filters*factor, kernel_size=old_layer.kernel_size,
                                                          strides=old_layer.strides,
                                                          padding='same',
                                                          activation=old_layer.activation,
                                                          name=name)
        if next_conv:
            modified_next_layer = keras.layers.Conv2D(next_layer.filters, kernel_size=next_layer.kernel_size,
                                                      strides=next_layer.strides,
                                                      padding=next_layer.padding,
                                                      activation=next_layer.activation,
                                                      name=next_layer.name)
        else:
            modified_next_layer = keras.layers.Conv2DTranspose(next_layer.filters, kernel_size=next_layer.kernel_size,
                                                               strides=next_layer.strides,
                                                               padding=next_layer.padding,
                                                               activation=next_layer.activation,
                                                               name=next_layer.name)

        new_model = self.reconnect_model(layer, [modified_layer, modified_next_layer], replace=True)

        # Get weights for knowledge transfer
        w0, b0 = old_layer.get_weights()
        w1, b1 = next_layer.get_weights()
        u0, v0 = new_model.layers[layer].get_weights()
        u1, v1 = new_model.layers[layer + 1].get_weights()
        u1 = u1.astype(np.float64)  # TODO: might not be needed

        # Copy the original filters
        if old_conv:
            u0[:, :, :, :n_filters] = w0
        else:
            u0[:, :, :n_filters, :] = w0

        v0[:n_filters] = b0
        if next_conv:
            u1[:, :, :n_filters, :] = w1
        else:
            u1[:, :, :, :n_filters] = w1

        # Select which original filters will be used for the new ones and copy them
        g = np.random.choice(n_filters, size=(n_filters*factor - n_filters))  # g function from the paper
        if old_conv:
            u0[:, :, :, n_filters:] = w0[:, :, :, g]
        else:
            u0[:, :, n_filters:, :] = w0[:, :, g, :]
        v0[n_filters:] = b0[g]
        if next_conv:
            u1[:, :, n_filters:, :] = w1[:, :, g, :]
        else:
            u1[:, :, :, n_filters:] = w1[:, :, :, g]

        # Normalize each new filter of (layer + 1) by how many times the source filter has been selected
        source = np.concatenate((np.arange(n_filters, dtype=np.int), g))
        # print 'source: {}'.format(source)  # TODO: remove prints
        counts = np.bincount(source)  # TODO: remove prints
        # print 'counts: {}'.format(counts)
        norm = counts[source].astype(np.float64)
        if next_conv:
            u1 = np.swapaxes(np.swapaxes(u1, 2, 3)/norm, 2, 3)
        else:
            u1 = u1 / norm

        # Update changes into model
        new_model.layers[layer].set_weights([u0, v0])
        new_model.layers[layer + 1].set_weights([u1, v1])
        new_model.compile(optimizer='rmsprop', loss='mean_squared_error')

        return Network(model=new_model)

    def wider(self, layer, factor=2):
        """
        Adds convolutional filters to a layer and adjusts weights using transfer learning
        :param layer: layer number that will be modified
        :param factor: value by which number of filters is increased
        :return: A copy of self with the updated layer
        """
        old_layer = self.model.layers[layer]
        old_layer_type = type(old_layer).__name__
        old_conv = old_layer_type == 'Conv2D'
        n_filters = old_layer.filters
        name = old_layer.name + '_wider'

        assert old_layer_type == 'Conv2D' or old_layer_type == 'Conv2DTranspose'

        if layer == (len(self.model.layers) - len(self.model.output) - 1):
            next_layers = self.model.layers[-4:]
            next_layers_type = ['Conv2D'] * 4
        else:
            next_layers = [self.model.layers[layer + 1]]
            next_layers_type = [type(next_layers[0]).__name__]


        # Replace old layer with a wider version and reconnect the graph
        modified_layers = []
        connectivity = [-1]
        if old_conv:
            modified_layers.append(keras.layers.Conv2D(n_filters*factor, kernel_size=old_layer.kernel_size,
                                                       strides=old_layer.strides,
                                                       padding='same',
                                                       activation=old_layer.activation,
                                                       name=name))
        else:
            modified_layers.append(keras.layers.Conv2DTranspose(n_filters*factor, kernel_size=old_layer.kernel_size,
                                                                strides=old_layer.strides,
                                                                padding='same',
                                                                activation=old_layer.activation,
                                                                name=name))
        for next_layer_idx, next_layer in enumerate(next_layers):
            connectivity.append(0)
            if next_layers_type[next_layer_idx] == 'Conv2D':
                modified_layers.append(keras.layers.Conv2D(next_layer.filters, kernel_size=next_layer.kernel_size,
                                                           strides=next_layer.strides,
                                                           padding=next_layer.padding,
                                                           activation=next_layer.activation,
                                                           name=next_layer.name))
            elif next_layers_type[next_layer_idx] == 'Conv2DTranspose':
                modified_layers.append(keras.layers.Conv2DTranspose(next_layer.filters, kernel_size=next_layer.kernel_size,
                                                                    strides=next_layer.strides,
                                                                    padding=next_layer.padding,
                                                                    activation=next_layer.activation,
                                                                    name=next_layer.name))
            else:
                raise TypeError('Expected next layer to be either \'Conv2D\' or \'Conv2DTranspose\' but received {} instead'.format(next_layers_type[next_layer_idx]))

        new_model = self.reconnect_model(layer, modified_layers, connectivity=connectivity, replace=True)

        # Widen top layer
        w0, b0 = old_layer.get_weights()
        u0, v0 = new_model.layers[layer].get_weights()

        g = np.random.choice(n_filters, size=(n_filters*factor - n_filters))  # g function from the paper

        if old_conv:
            u0[:, :, :, :n_filters] = w0
        else:
            u0[:, :, :n_filters, :] = w0

        v0[:n_filters] = b0

        if old_conv:
            u0[:, :, :, n_filters:] = w0[:, :, :, g]
        else:
            u0[:, :, n_filters:, :] = w0[:, :, g, :]
        v0[n_filters:] = b0[g]

        new_model.layers[layer].set_weights([u0, v0])  # Update weights

        # To be used in filter normalization
        source = np.concatenate((np.arange(n_filters, dtype=np.int), g))
        counts = np.bincount(source)
        norm = counts[source].astype(np.float64)

        # Widen bottom layer(s)
        for next_layer_idx, next_layer in enumerate(next_layers):
            new_next_layer = new_model.layers[layer]._outbound_nodes[next_layer_idx].outbound_layer

            w1, b1 = next_layer.get_weights()
            u1, v1 = new_next_layer.get_weights()
            is_conv = next_layers_type[next_layer_idx] == 'Conv2D'

            if is_conv:
                u1[:, :, :n_filters, :] = w1
            else:
                u1[:, :, :, :n_filters] = w1

            if is_conv:
                u1[:, :, n_filters:, :] = w1[:, :, g, :]
            else:
                u1[:, :, :, n_filters:] = w1[:, :, :, g]

            # Normalize each new filter of (layer + 1) by how many times the source filter has been selected
            if is_conv:
                u1 = np.swapaxes(np.swapaxes(u1, 2, 3)/norm, 2, 3)
            else:
                u1 = u1 / norm

            # Update changes into model
            new_next_layer.set_weights([u1, v1])

        # Make the model ready for training
        new_model.compile(optimizer='rmsprop', loss='mean_squared_error')

        return Network(model=new_model)

    def deeper(self, layer):
        """
        Adds a new layer and adjusts weights using transfer learning
        :param layer: layer number bellow which the new layer will be added
        :return: A copy of self with the updated layers
        """
        temp_model = self.copy_model()
        input_layer = temp_model.layers[layer]
        input_layer_type = type(input_layer).__name__
        assert input_layer_type in ['Conv2D', 'Conv2DTranspose']
        input_layer_conv = input_layer_type == 'Conv2D'
        kernel_size = input_layer.kernel_size
        assert (np.mod(kernel_size, 2) == 1).all()
        n_filters = input_layer.filters
        activation = input_layer.activation
        new_name = input_layer.name + '_deeper_{}'.format(time.time())

        if input_layer_conv:
            new_layer = keras.layers.Conv2D(n_filters, kernel_size=kernel_size, padding='same', activation=activation,
                                            kernel_initializer='zeros',
                                            bias_initializer='zeros',
                                            name=new_name)
        else:
            new_layer = keras.layers.Conv2DTranspose(n_filters, kernel_size=kernel_size, padding='same', activation=activation,
                                            kernel_initializer='zeros',
                                            bias_initializer='zeros',
                                            name=new_name)

        new_model = self.reconnect_model(layer, [new_layer])

        w, b = new_model.layers[layer + 1].get_weights()
        center_row = int(kernel_size[0]/2.)
        center_col = int(kernel_size[1]/2.)
        for f in range(n_filters):
            w[center_row, center_col, f, f] = 1.
        new_model.layers[layer + 1].set_weights([w, b])
        new_model.compile(optimizer='rmsprop', loss='mean_squared_error')

        return Network(model=new_model)

    def raw_predict(self, depth, subtract_mean=True):
        """
        Same as predict but the angles are not postprocessed. For visualization purposes only
        """
        assert isinstance(depth, np.ndarray)
        assert depth.ndim in [2, 3, 4]
        assert isinstance(subtract_mean, bool)

        if depth.ndim == 2:
            depth = depth.reshape((1,) + depth.shape + (1,))
        elif depth.ndim == 3:
            depth = depth.reshape(depth.shape + (1,))
        assert depth.shape[1:3] == (self.width, self.height), 'Invalid input shape'

        if subtract_mean:
            flat_depth = depth.flatten() - depth.mean(axis=(1, 2)).repeat(depth.shape[1] * depth.shape[2])
            depth = flat_depth.reshape(depth.shape)

        return self.model.predict(depth, batch_size=24)

    def predict(self, depth, subtract_mean=True):
        """
        Uses model to evaluate depth image.

        Args
        ----
        depth: np.ndarray, required
            A ndarray with the depth image(s) to be evaluated. May or may not
            contain singletons (e.g. shape(heigth, width) and shape(N,height, width,1)
            are both valid).
        subtract_mean: boolean, optional
            If true each image will be centered around its mean

        Returns
        -------
        output: np.ndarray
            A tuple of length 3 (position, angle, width). Each element is a
            ndarray of shape (n, height, width, 1)
        """
        assert isinstance(depth, np.ndarray)
        assert depth.ndim in [2,3,4]
        assert isinstance(subtract_mean, bool)

        if depth.ndim == 2:
            depth = depth.reshape((1,) + depth.shape + (1,))
        elif depth.ndim == 3:
            depth = depth.reshape(depth.shape + (1,))
        assert depth.shape[1:3] == (self.width, self.height), 'Invalid input shape'

        if subtract_mean:
            flat_depth = depth.flatten() - depth.mean(axis=(1,2)).repeat(depth.shape[1]*depth.shape[2])
            depth = flat_depth.reshape(depth.shape)

        model_output = self.model.predict(depth, batch_size=24)
        position = model_output[0]
        angle = np.arctan2(model_output[2], model_output[1])/2.0
        width = model_output[3] * 150.

        return position, angle, width

    def train(self, train_generator, batch_sz, n_epochs, verbose=1, callbacks=None, test_generator=None):
        validation_steps = None if test_generator is None else test_generator.n_samples // batch_sz
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_generator.n_samples // batch_sz,
                                 epochs=n_epochs,
                                 verbose=verbose,
                                 validation_data=test_generator,
                                 validation_steps=validation_steps,
                                 shuffle=True,
                                 callbacks=callbacks)

    def predict_pointcloud(self, pcl, remove_plane=True):
        pass


if __name__ == '__main__':
    network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
    rgbs, depths = read_input_from_scene_path('/Users/mario/Desktop/scenes', network.height, network.width)
    positions, angles, widths = network.predict(depths)
    for img_idx in range(depths.shape[0]):
        save_output_plot(depths[img_idx], positions[img_idx],
                angles[img_idx], widths[img_idx], '/Users/mario/Desktop/results/{}.png'.format(img_idx))


