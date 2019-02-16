import os
import glob
from random import shuffle
import argparse

import numpy as np
import h5py

from keras.models import load_model

import matplotlib.pyplot as plt
from skimage.filters import gaussian

from ggcnn.dataset_processing.grasp import BoundingBoxes, detect_grasps

# Networks to test.
NETWORK = 'ggcnn/data/networks/ggcnn_rss'  # glob synatx to output network folders.
EPOCH = None  # Specify epoch or None to test all.

WRITE_LOG = True
LOGFILE = 'evaluation_output.txt'

NO_GRASPS = 1  # Number of local maxima to check against ground truth grasps.
EXPORT_VISUALIZATION = True
VISUALISE_FAILURES = True
VISUALISE_SUCCESSES = True
EXPORT_PATH = 'data/evaluation/{}/{}'
EXPORT_FN = os.path.join(EXPORT_PATH,'{}.png')


def write_log(s):
    if WRITE_LOG:
        with open(LOGFILE, 'a') as f:
            f.write(s)


def plot_output(name, rgb_img, depth_img, grasp_position_img, grasp_angle_img, ground_truth_bbs, no_grasps=1, grasp_width_img=None):
        """
        Visualise the outputs.
        """
        if not EXPORT_VISUALIZATION:
            return
        grasp_position_img = gaussian(grasp_position_img, 5.0, preserve_range=True)

        if grasp_width_img is not None:
            grasp_width_img = gaussian(grasp_width_img, 1.0, preserve_range=True)

        gt_bbs = BoundingBoxes.load_from_array(ground_truth_bbs)
        gs = detect_grasps(grasp_position_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps, ang_threshold=0)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(rgb_img)
        for g in gs:
            g.plot(ax)

        for g in gt_bbs:
            g.plot(ax, color='g')

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img)
        for g in gs:
            g.plot(ax, color='r')

        for g in gt_bbs:
            g.plot(ax, color='g')

        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(grasp_position_img, cmap='Reds', vmin=0, vmax=1)

        ax = fig.add_subplot(2, 2, 4)
        plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        plt.colorbar(plot)
        plt.savefig(name)
        plt.close(fig)


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


def run(model_path, dataset_fn, export, epochs):
    global NO_GRASPS, VISUALISE_FAILURES, VISUALISE_SUCCESSES

    # Load the dataset data.
    model_folders = glob.glob(model_path)
    model_folders.sort()

    for model_folder in model_folders:

        model_name = model_folder.split('/')[-1]
        print('Evaluating:  %s, epoch %s' % (model_folder, EPOCH))

        write_log('\n')
        write_log(model_name)
        write_log('\t')

        #dataset_fn = ''
        #with open(os.path.join(model_folder, '_dataset.txt')) as f:
        #    dataset_fn = f.readline()
        #    if dataset_fn[-1] == '\n':
        #        dataset_fn = dataset_fn[:-1]

        #f = h5py.File(dataset_fn, 'r')
        f = h5py.File(dataset_fn)

        img_ids = np.array(f['test/img_id'])
        rgb_imgs = np.array(f['test/rgb'])
        depth_imgs = np.array(f['test/depth_inpainted'])
        bbs_all = np.array(f['test/bounding_boxes'])

        f.close()

        epochs = [EPOCH]
        if EPOCH is None:
            # Get all of them
            saved_models = glob.glob(os.path.join(model_folder, 'epoch_*_model.hdf5'))
            saved_models.sort()
            epochs = [int(s[-13:-11]) for s in saved_models]

        for epoch in epochs:
            epoch_name = model_name + '_{}'.format(epoch)
            if EXPORT_VISUALIZATION:
                if not os.path.exists(EXPORT_PATH.format(epoch_name, 'success')):
                    os.makedirs(EXPORT_PATH.format(epoch_name, 'success'))
                if not os.path.exists(EXPORT_PATH.format(epoch_name, 'failure')):
                    os.makedirs(EXPORT_PATH.format(epoch_name, 'failure'))
            # Load the output data.
            model_output_fn = os.path.join(model_folder, 'epoch_%02d_val_output.npz' % epoch)
            if os.path.exists(model_output_fn):
                # Check if there's a pre-computed output.
                model_output_data = np.load(model_output_fn)
                grasp_positions_out = model_output_data['pos_out']
                grasp_angles_out = model_output_data['angle_out']
                grasp_width_out = model_output_data['grasp_width_out']
            else:
                # Load the model and compute the output.
                print('No pre-computed values.  Computing now.')
                model_checkpoint_fn = os.path.join(model_folder, 'epoch_%02d_model.hdf5' % epoch)
                model = load_model(model_checkpoint_fn)
                #input_data_fn = os.path.join(model_folder, '_val_input.npy')
                #input_data = np.load(input_data_fn)
                input_data = np.expand_dims(depth_imgs.copy(), axis=3)
                model_output_data = model.predict(input_data)
                grasp_positions_out = model_output_data[0]
                grasp_angles_out = np.arctan2(model_output_data[2], model_output_data[1])/2.0
                grasp_width_out = model_output_data[3] * 150.0

            # IOU TESTING.
            succeeded, failed = calculate_iou_matches(grasp_positions_out, grasp_angles_out, bbs_all, no_grasps=NO_GRASPS, grasp_width_out=grasp_width_out)

            s = len(succeeded) * 1.0
            f = len(failed) * 1.0
            print('%s\t%s\t%s/%s\t%0.02f%s' % (model_folder.split('/')[-1], epoch, s, s+f, s/(s+f)*100.0, '%'))
            write_log('%0.02f\t' % (s/(s+f)*100.0))

            if VISUALISE_FAILURES:
                print('Plotting Failures')
                shuffle(failed)
                for i in failed:
                    plot_output(EXPORT_FN.format(epoch_name, 'failure', i), rgb_imgs[i, ], depth_imgs[i, ], grasp_positions_out[i, ].squeeze(), grasp_angles_out[i, ].squeeze(), bbs_all[i, ],
                                no_grasps=NO_GRASPS, grasp_width_img=grasp_width_out[i, ].squeeze())

            if VISUALISE_SUCCESSES:
                print('Plotting Successes')
                shuffle(succeeded)
                for i in succeeded:
                    plot_output(EXPORT_FN.format(epoch_name, 'success', i), rgb_imgs[i, ], depth_imgs[i, ], grasp_positions_out[i, ].squeeze(), grasp_angles_out[i, ].squeeze(), bbs_all[i, ],
                                no_grasps=NO_GRASPS, grasp_width_img=grasp_width_out[i, ].squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model folder(s)')
    parser.add_argument('dataset', help='path to dataset file')
    parser.add_argument('-o', action='store_true', help='export success and failure visualization')
    parser.add_argument('-e', nargs='?', default=None, help='export success and failure visualization')

    args = parser.parse_args()
    global EXPORT_VISUALIZATION
    EXPORT_VISUALIZATION = args.o

    run(args.model, args.dataset, args.o, args.e)
