from keras.models import load_model
from ggcnn.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

class Network:
    def __init__(self, model_fn):
        self.model = load_model(model_fn)

    @property
    def height(self):
        return self.model.input.shape[1]

    @property
    def width(self):
        return self.model.input.shape[2]

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

        model_output = self.model.predict(depth)
        position = model_output[0]
        angle = np.arctan2(model_output[2], model_output[1])/2.0
        width = model_output[3] * 150.

        return position, angle, width

    def _get_output_plot(self, depth, position, angle, width, no_grasps=1):
        depth = depth.squeeze()
        angle = angle.squeeze()
        position = gaussian(position.squeeze(), 5.0, preserve_range=True)
        width = gaussian(width.squeeze(), 1.0, preserve_range=True)

        gs = detect_grasps(position, angle, width_img=width, no_grasps=no_grasps, ang_threshold=0)

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(1, 3, 1)
        ax.set_title('Depth')
        ax.imshow(depth)
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

    def plot_output(self, depth, positions, angles, widths, no_grasps=1):
        fig = self._get_output_plot(depth, positions, angles, widths, no_grasps)
        plt.show(fig)

    def save_output_plot(self, depth, positions, angles, widths, filename, no_grasps=1):
        self._get_output_plot(depth, positions, angles, widths, no_grasps)
        plt.savefig(filename)

    def read_input_from_scene_path(self, scene_path):
        input_fns = glob.glob(os.path.join(scene_path, '*.npy'))
        assert len(input_fns) > 0, 'No input files were found'

        return self.read_input(input_fns)

    def read_input(self, input_fns):
        """
        Reads .npy files in input_fns and combines them

        Args
        ----
            input_fns: list of str
                list of .npy files
        Returns
        -------
            rgb: np.ndarray
                a (N,height, width, 3) array of type uint8  encoding rgb images
            depth: np.nadarray
                a (N, height, width) array of type float32 encoding the depth
                images
        """
        index = {}
        input = np.zeros((len(input_fns), self.height, self.width, 4), np.float32)

        for idx, input_fn in enumerate(input_fns):
            input_name = input_fn.split('/')[-1].split('_')[0]
            index[input_name] = idx
            input[idx,:] = np.load(input_fn)

        return input[:,:,:,0:3].astype(np.uint8), input[:,:,:,3]

if __name__ == '__main__':
    network = Network('ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
    rgbs, depths = network.read_input_from_scene_path('simulator/scenes')
    positions, angles, widths = network.predict(depths)
    for img_idx in range(depths.shape[0]):
        network.save_output_plot(depths[img_idx], positions[img_idx],
                angles[img_idx], widths[img_idx], '{}.png'.format(img_idx))

