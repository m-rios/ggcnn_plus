import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from ggcnn.dataset_processing.grasp import detect_grasps, BoundingBoxes
from simulator.simulator import Simulator

SCENES_PATH = os.environ['GGCNN_SCENES_PATH']
SHAPENET_PATH = os.environ['SHAPENET_PATH']

def read_input(input_fns, width, height, ch=4):
    index = {}
    input = np.zeros((len(input_fns), height, width, ch), np.float32)

    for idx, input_fn in enumerate(input_fns):
        input_name = input_fn.split('/')[-1].split('_')[0]
        index[input_name] = idx
        input[idx,:] = np.load(input_fn)

    return index, input

def plot_output(name, rgb_img, depth_img, grasp_position_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
        """
        Visualise the outputs.
        """
        import ipdb; ipdb.set_trace() # BREAKPOINT
        grasp_position_img = gaussian(grasp_position_img, 5.0, preserve_range=True)

        if grasp_width_img is not None:
            grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)

        gs = detect_grasps(grasp_position_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, ang_threshold=0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('RGB')
        ax.imshow(rgb_img)
        for g in gs:
            g.plot(ax, 'r')

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('Depth')
        ax.imshow(depth_img)
        for g in gs:
            g.plot(ax, 'r')

        ax = fig.add_subplot(2, 2, 3)
        ax.set_title('Grasp quality')
        ax.imshow(grasp_position_img, cmap='Reds', vmin=0, vmax=1)

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('Grasp angle')
        plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        plt.colorbar(plot)
        plt.savefig(name)
        plt.close(fig)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the root directory of a model')
    parser.add_argument('-e', nargs='+', default=None, type=int, help='epochs to evaluate, if next arg is model, separate with -- ')
    parser.add_argument('-v', action='store_true', help='visualize model output')

    args = parser.parse_args()


    model_fns = glob.glob(args.model + '/*.hdf5')

    assert len(model_fns) > 0

    # Get input size and initialize simulator camera with it
    from keras.models import load_model
    _, height, width, _ = load_model(model_fns[0]).input_shape

    sim = Simulator()
    sim.cam.height = height
    sim.cam.width = width

    input_fns = glob.glob(SCENES_PATH+'/*_{}_{}.npy'.format(width, height))
    scene_fns = glob.glob(SCENES_PATH + '/*.csv')

    assert len(input_fns) >= len(scene_fns), 'Missing input images of size ({},{})'.format(width, height)

    # Iterate through epochs
    for model_fn in model_fns:
        model_name = model_fn.split('/')[-2]
        epoch = int(model_fn.split('_')[-2])

        if args.e is not None and epoch not in args.e:
            continue

        print('Evaluating epoch {} model {}'.format(model_name, epoch))

        model = load_model(model_fn)
        input_idx, input_imgs = read_input(input_fns, width, height)
        rgb = input_imgs[:,:,:,0:3].astype(np.uint8)
        depth = np.expand_dims(input_imgs[:,:,:,3], 3)

        model_output_data = model.predict(depth)
        grasp_positions_out = model_output_data[0]
        grasp_angles_out = np.arctan2(model_output_data[2], model_output_data[1])/2.0
        grasp_width_out = model_output_data[3] * 150.0

        # Test results for each scene
        for scene_fn in scene_fns:
            sim.restore(scene_fn, SHAPENET_PATH)
            # Compute grasp 6DOF coordiantes w.r.t camera frame
            # Send grasp to simulator and evaluate


