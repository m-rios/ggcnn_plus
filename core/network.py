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
        if model is not None:
            self.model = model
        elif model_fn is not None:
            self.model = keras.models.load_model(model_fn)
        else:
            raise ValueError('Either model_fn or model must be provided')

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

    def copy_model(self):
        new_model = keras.models.clone_model(self.model)
        new_model.set_weights(self.model.get_weights())
        return new_model

    def reconnect_model(self, layer_idx, layers, replace=False):
        """
        Reconnects last_layer with the reminder of layers after layer_idx
        :param layer_idx: Index in the model of the first layer to be reconnected
        :param layers: An ordered list of the layers to be reconnected
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

    def wider(self, layer, factor=2):
        """
        Adds convolutional filters to a layer and adjusts weights using transfer learning
        :param layer: layer number that will be modified
        :param factor: value by which number of filters is increased
        :return: A copy of self with the updated layer
        """
        old_layer = self.model.layers[layer]
        next_layer = self.model.layers[layer + 1]
        next_layer_type = type(next_layer).__name__
        next_conv = next_layer_type == 'Conv2D'
        n_filters = old_layer.filters
        name = old_layer.name + '_wider'

        assert next_layer_type == 'Conv2D' or next_layer_type == 'Conv2DTranspose'

        # Replace old layer with a wider version and reconnect the graph
        modified_layer = keras.layers.Conv2D(n_filters*factor, kernel_size=old_layer.kernel_size,
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
        u0[:, :, :, :n_filters] = w0
        v0[:n_filters] = b0
        if next_conv:
            u1[:, :, :n_filters, :] = w1
        else:
            u1[:, :, :, :n_filters] = w1

        # Select which original filters will be used for the new ones and copy them
        g = np.random.choice(n_filters, size=(n_filters*factor - n_filters))  # g function from the paper
        u0[:, :, :, n_filters:] = w0[:, :, :, g]
        v0[n_filters:] = b0[g]
        if next_conv:
            u1[:, :, n_filters:, :] = w1[:, :, g, :]
        else:
            u1[:, :, :, n_filters:] = w1[:, :, :, g]

        # Normalize each new filter of (layer + 1) by how many times the source filter has been selected
        source = np.concatenate((np.arange(n_filters, dtype=np.int), g))
        print 'source: {}'.format(source)
        counts = np.bincount(source)
        print 'counts: {}'.format(counts)
        norm = counts[source].astype(np.float64)
        if next_conv:
            u1 = np.swapaxes(np.swapaxes(u1, 2, 3)/norm, 2, 3)
        else:
            u1 = u1 / norm

        # Update changes into model
        new_model.layers[layer].set_weights([u0, v0])
        new_model.layers[layer + 1].set_weights([u1, v1])

        return Network(model=new_model)

    def deeper(self, layer):
        """
        Adds a new layer and adjusts weights using transfer learning
        :param layer: layer number bellow which the new layer will be added
        :return: A copy of self with the updated layers
        """
        temp_model = self.copy_model()
        input_layer = temp_model.layers[layer]
        assert type(input_layer).__name__ == 'Conv2D'
        kernel_size = input_layer.kernel_size
        assert (np.mod(kernel_size, 2) == 1).all()
        n_filters = input_layer.filters
        activation = input_layer.activation
        new_name = input_layer.name + '_deeper_{}'.format(time.time())

        new_layer = keras.layers.Conv2D(n_filters, kernel_size=kernel_size, padding='same', activation=activation,
                                        kernel_initializer='zeros',
                                        bias_initializer='zeros',
                                        name=new_name)
        # new_model = self.insert_tensor(layer, x)
        new_model = self.reconnect_model(layer, [new_layer])

        w, b = new_model.layers[layer + 1].get_weights()
        center_row = int(kernel_size[0]/2.)
        center_col = int(kernel_size[1]/2.)
        for f in range(n_filters):
            w[center_row, center_col, f, f] = 1.
        new_model.layers[layer + 1].set_weights([w, b])

        return Network(model=new_model)

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


if __name__ == '__main__':
    network = Network(model_fn='../ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
    rgbs, depths = read_input_from_scene_path('/Users/mario/Desktop/scenes', network.height, network.width)
    positions, angles, widths = network.predict(depths)
    for img_idx in range(depths.shape[0]):
        save_output_plot(depths[img_idx], positions[img_idx],
                angles[img_idx], widths[img_idx], '/Users/mario/Desktop/results/{}.png'.format(img_idx))


