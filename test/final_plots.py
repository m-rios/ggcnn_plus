import pylab as plt
import numpy as np
import h5py
import pandas as pd
from ggcnn.dataset_processing.grasp import BoundingBoxes
from unittest import TestCase
from tabulate import tabulate

DEBUG = True
OUTPUT_PATH = '/Users/mario/Dropbox/Apps/Texpad/Master thesis/Thesis/figures/'
RESULTS_PATH = '/Users/mario/Developer/msc-thesis/data/results/'

gray = '#888888'
red = '#f20000'

dash = [8, 2]
dot_dash = [1, 2, 8, 2]

linewidth = 1.2

reported_ggcnn_adv = 0.84  # The results they published in real world open loop experiment
reported_ggcnn_house = 0.92  # The results they published in real world open loop experiment
vanilla_ggcnn = 0.785  # The results of their published network on our simulator


def _read_simulation(sim_results_fn):
    sim_epochs = []
    sim_values = []
    with open(sim_results_fn, 'r') as f:
        for line in f.readlines():
            fields = line.split()
            if len(fields) == 0 or fields[0] != 'Epoch':
                continue

            sim_epochs.append(fields[1].replace(':', ''))
            sim_values.append(fields[2].replace('%', ''))
    sim_epochs = np.array(sim_epochs).astype(np.int)
    sim_values = np.array(sim_values).astype(np.float)
    sort_idx = np.argsort(sim_epochs)
    sim_epochs = sim_epochs[sort_idx]
    sim_values = sim_values[sort_idx]

    return sim_epochs, sim_values


def _read_iou(iou_results_fn):
    with open(iou_results_fn, 'r') as f:
        iou_values = f.readline().split()[1:]

    values = np.array(iou_values).astype(np.float) / 100.
    epochs = range(1, len(values) + 1)

    return epochs, values


def _read_beam(beam_results_fn):
    iou_values = []

    with open(beam_results_fn, 'r') as f:
        for line in f.readlines():
            fields = line.split()
            if len(fields) == 0 or fields[0] != 'Node:':
                continue
            iou_values.append(fields[3])

    epochs = range(len(iou_values))
    iou_values = np.array(iou_values).astype(np.float)

    return epochs, iou_values


def _save_plot(filename):
    if not DEBUG:
        plot_fn = OUTPUT_PATH + filename + '.eps'
        plt.savefig(plot_fn)
        print 'saved plot as {}'.format(plot_fn)


def _parse_orthonet_results(fn):
    f = open(fn)
    metadata = {}
    # Skip headers
    line = f.readline()
    while len(line) < 11 or line[:11] != 'scene_name,':
        fields = line.split(':')
        metadata[fields[0]] = None if len(fields) < 2 else ','.join(fields[1:])
        line = f.readline()
    line = f.readline()

    results = []
    errors = 0
    while line:
        if line.find('failed due to exception') > 0:
            line = f.readline()
            continue

        fields = line.split(',')

        if len(fields) != 7:
            line = f.readline()
            continue

        results.append([
            fields[0],             # scene_name
            np.array(fields[1]),   # p
            np.array(fields[2]),   # z
            np.array(fields[3]),   # x
            float(fields[4]),      # w
            fields[5],             # view
            fields[6] == 'True\n'  # success
        ])

        line = f.readline()

    return metadata, results, errors

# This is just so I can easily run an individual method from the IDE
class FinalPlots(TestCase):

    def test_simulator_baseline(self):
        iou_results_fn = '/Users/mario/Developer/msc-thesis/data/results/190503_1946__ggcnn_9_5_3__32_16_8/iou/iou_cornell_25.txt'
        # sim_results_fn = '/Users/mario/Developer/msc-thesis/data/results/190922_1527_ggcnn_cornell/results.txt'
        sim_results_fn = '/Users/mario/Developer/msc-thesis/data/results/190925_2003_ggcnn_cornell/results.txt'

        sim_epochs, sim_values = _read_simulation(sim_results_fn)
        iou_epochs, iou_values = _read_iou(iou_results_fn)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        # plt.title('Simulation Baseline')
        ax1.set_title('Simulation')
        ax1.plot(range(1, 51), np.tile(vanilla_ggcnn, 50), color='k', linewidth=linewidth)
        ax1.plot(sim_epochs, sim_values, color='k', marker='.')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('% successful grasps')
        ax1.legend(['GGCNN_sim', 'simulation'])
        ax1.grid(True)

        # ax2 = ax1.twinx()
        ax2.set_title('IOU')
        ax2.plot(iou_epochs, iou_values, color='k', marker='.', linewidth=linewidth)
        ax2.set_ylabel('% IOU > 0.25', color='k')
        ax2.set_xlabel('Epoch')
        ax2.grid(True)

        _save_plot('simulation_baseline')

    def test_jacquard_baseline(self):
        plt.figure()
        jacquard = '/Users/mario/Developer/msc-thesis/data/results/190402_0535__ggcnn_9_5_3__32_16_8/'
        jacquard = jacquard + 'results.txt'
        cornell = '/Users/mario/Developer/msc-thesis/data/results/190503_1946__ggcnn_9_5_3__32_16_8/'
        cornell = cornell + 'results.txt'

        j_epochs, j_values = _read_simulation(jacquard)
        c_epochs, c_values = _read_simulation(cornell)

        plt.plot(j_epochs, j_values, '-k.')
        plt.plot(c_epochs, c_values, '-r.')
        plt.legend(['Jacquard', 'Cornlell'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        _save_plot('jacquard_baseline')

    def test_beam_search_improve(self):
        plt.figure(figsize=(12, 5))
        iou_paths = [
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/',
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_5/',
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_10/',
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_2+8/',
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_transpose/',
        ]

        sim_paths = [
            # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_2/',
            # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_5/',
            # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_10/',
            # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_2+8/',
            # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_last/'
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_2/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_5/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_10/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_2+8/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_last/'
        ]

        # colors = [red, 'g', 'b', 'k']
        markers = ['.'] * len(iou_paths)

        plt.subplot(121)
        for idx, iou_path in enumerate(iou_paths):
            iou_results_fn = iou_path + 'results.txt'

            epochs, iou_values = _read_beam(iou_results_fn)
            plt.plot(epochs, iou_values, marker=markers[idx], linewidth=linewidth)

        plt.title('IOU')
        plt.xlabel('Depth')
        plt.ylabel('Accuracy')
        plt.legend(['e=2', 'e=5', 'e=10', 'e=2 r = 8', 'e=2 transpose'], markerscale=0)

        plt.subplot(122)
        for idx, sim_path in enumerate(sim_paths):
            sim_results_fn = sim_path + 'results.txt'

            sim_epochs, sim_values = _read_simulation(sim_results_fn)
            sim_epochs = np.insert(sim_epochs, 0, -1) + 1
            sim_values = np.insert(sim_values, 0, vanilla_ggcnn)

            plt.plot(sim_epochs, sim_values, marker=markers[idx], dashes=dash, linewidth=linewidth)

        plt.title('SIM')
        plt.xlabel('Depth')
        plt.legend(['e=2', 'e=5', 'e=10', 'e=2 r = 8', 'e=2 transpose'], markerscale=0)

        _save_plot('beam_search_improve')

    def test_beam_search_optimize(self):
        plt.figure()
        iou_paths = [
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_narrow/',
            '/Users/mario/Developer/msc-thesis/data/networks/beam_search_shallow/',
        ]

        sim_paths = [
            # '/Users/mario/Developer/msc-thesis/data/results/190917_1928_beam_search_narrow/',
            # '/Users/mario/Developer/msc-thesis/data/results/190917_1928_beam_search_190625_2054/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_narrow/',
            '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_190625_2054/',
        ]

        starting_values = [0.64, 0.65]
        colors = [red, 'k']
        marker = '.'

        for idx, iou_path in enumerate(iou_paths):
            iou_results_fn = iou_path + 'results.txt'

            epochs, iou_values = _read_beam(iou_results_fn)

            if idx == 0:
                epochs = np.delete(epochs, -1)
                iou_values = np.delete(iou_values, -1)

            plt.plot(epochs, iou_values, color=colors[idx], marker=marker, linewidth=linewidth)

        for idx, sim_path in enumerate(sim_paths):
            sim_results_fn = sim_path + 'results.txt'

            sim_epochs, sim_values = _read_simulation(sim_results_fn)
            sim_epochs = np.insert(sim_epochs, 0, -1) + 1
            sim_values = np.insert(sim_values, 0, starting_values[idx])

            if idx == 0:
                sim_epochs = np.delete(sim_epochs, -1)
                sim_values = np.delete(sim_values, -1)

            plt.plot(sim_epochs, sim_values, color=colors[idx], marker=marker, dashes=dash, linewidth=linewidth)

        plt.legend(['narrow IOU', 'shallow IOU', 'narrow SIM', 'shallow SIM'], markerscale=0)
        plt.xlabel('Depth')
        plt.ylabel('Accuracy')
        plt.title('Optimizing')
        _save_plot('beam_search_optimize')

    def test_beam_loss(self):

        loss_paths = [
            RESULTS_PATH + 'beam_search_190926_1950/',
            RESULTS_PATH + 'beam_search_190926_2111/',
            RESULTS_PATH + 'beam_search_190928_1316/',
        ]

        sim_paths = [
            RESULTS_PATH + '190929_0958_beam_search_190926_1950/',
            RESULTS_PATH + '190929_1125_beam_search_190926_2111/',
            RESULTS_PATH + '190929_1051_beam_search_190928_1316/',
        ]

        colors = [red, 'k', 'b']
        marker = '.'
        plt.subplot(121)
        for idx, loss_path in enumerate(loss_paths):
            loss_results_fn = loss_path + 'results.txt'

            epochs, loss_values = _read_beam(loss_results_fn)

            # if idx == 0:
            #     epochs = np.delete(epochs, -1)
            #     loss_values = np.delete(loss_values, -1)

            plt.plot(epochs, loss_values, color=colors[idx], marker=marker, linewidth=linewidth)
        plt.xlabel('Depth')
        plt.ylabel('Loss')
        plt.legend(['e=2', 'e=2 w/o lookahead', 'e=5 w/o lookahead'], markerscale=0)

        plt.subplot(122)
        for idx, sim_path in enumerate(sim_paths):
            sim_results_fn = sim_path + 'results.txt'

            sim_epochs, sim_values = _read_simulation(sim_results_fn)
            # sim_epochs = np.insert(sim_epochs, 0, -1) + 1
            # sim_values = np.insert(sim_values, 0, starting_values[idx])

            if idx == 0:
                sim_epochs = np.insert(sim_epochs, 0, -1) + 1
                sim_values = np.insert(sim_values, 0, 0.78)

            plt.plot(sim_epochs, sim_values, color=colors[idx], marker=marker, linewidth=linewidth)

        plt.legend(['e=2', 'e=2 w/o lookahead', 'e=5 w/o lookahead'], markerscale=0)
        plt.xlabel('Depth')
        plt.ylabel('% Successful grasps')
        plt.suptitle('Beam Search loss as heuristic')

    def test_input_outputs():
        from core.network import Network
        from skimage.filters import gaussian
        from ggcnn.dataset_processing.grasp import detect_grasps
        from ggcnn.dataset_processing.grasp import BoundingBoxes, BoundingBox
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ds = h5py.File('/Users/mario/Developer/msc-thesis/data/scenes/shapenetsem40_5.hdf5', 'r')
        depth = ds['depth'][31]
        net = Network('/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5')
        # net = Network('/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')
        pos, cos_im, sin_im, wid = net.raw_predict(depth)

        # pos = gaussian(pos.squeeze(), 5.0, preserve_range=True)
        # wid = gaussian(wid.squeeze(), 5.0, preserve_range=True)
        pos = pos.squeeze()
        cos_im = cos_im.squeeze()
        sin_im = sin_im.squeeze()
        wid = wid.squeeze()

        plt.figure('pos')
        ax = plt.gca()
        im = plt.imshow(pos)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[np.min(pos[:]), np.max(pos[:])], orientation='horizontal')
        # cbar.ax.set_xticklabels(['0', np.max(pos[:]).astype(np.str)])

        plt.figure('cos')
        ax = plt.gca()
        im = plt.imshow(cos_im)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[np.min(cos_im[:]), 0, np.max(cos_im[:])], orientation='horizontal')
        # cbar.ax.set_xticklabels([np.min(cos_im[:]).astype(np.str), np.max(cos_im[:]).astype(np.str)])

        plt.figure('sin')
        ax = plt.gca()
        im = plt.imshow(sin_im)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[np.min(sin_im[:]), 0, np.max(sin_im[:])], orientation='horizontal')
        # cbar.ax.set_xticklabels([-1, 0, 1])

        plt.figure('wid')
        ax = plt.gca()
        im = plt.imshow(wid)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ticks=[np.min(wid[:]), np.max(wid[:])], orientation='horizontal')
        # cbar.ax.set_xticklabels(['0', '74'])

        plt.figure()
        plt.imshow(depth)
        plt.axis('off')

    def test_jacquard_vs_cornell_dataset():
        j = h5py.File('/Users/mario/Developer/msc-thesis/data/datasets/preprocessed/jacquard_samples.hdf5', 'r')['train']
        # c = h5py.File('')

        bbs = BoundingBoxes.load_from_array(j['bounding_boxes'][2])

        plt.figure(figsize=(5, 2))
        plt.subplot(251)
        plt.imshow(j['depth_inpainted'][2])
        plt.title('Depth')

        plt.axis('off')
        plt.subplot(252)
        plt.imshow(j['grasp_points_img'][2])
        plt.title('Grasp Quality')
        plt.axis('off')
        plt.subplot(253)
        plt.imshow(j['grasp_width'][2])
        plt.title('Width')
        plt.axis('off')
        plt.subplot(254)
        plt.imshow(np.cos(2 * j['angle_img'][2]))
        plt.title('cos')
        plt.axis('off')
        plt.subplot(255)
        plt.imshow(np.sin(2 * j['angle_img'][2]))
        plt.title('sin')
        plt.axis('off')

    def test_orthonet_table(self):
        res_fns = [
            '/Users/mario/Developer/msc-thesis/data/results/orthonet_200224_184452_9731206.txt',
            '/Users/mario/Developer/msc-thesis/data/results/orthonet_200224_184456_9731208.txt',
            '/Users/mario/Developer/msc-thesis/data/results/orthonet_200224_184552_9731205.txt',
            '/Users/mario/Developer/msc-thesis/data/results/orthonet_200224_191010_9731395.txt',
        ]
        row = []
        for res_fn in res_fns:
            metadata, res, errors = _parse_orthonet_results(res_fn)

            network_name = 'network' in metadata and metadata['network'].split('/')[-2]
            angle = 'angle' in metadata and metadata['angle'].replace('\n', '')
            scoring = 'scoring' in metadata and metadata['scoring'].replace('\n', '')
            description = '%s %s %s' % (network_name, angle, scoring)

            success = np.sum(np.array(map(lambda x: x[6], res)))
            accuracy = success / float(len(res))

            top = filter(lambda x: x[5] == 'top', res)
            top_accuracy = np.sum(np.array(map(lambda x: x[6], top))) / float(len(top))
            n_top = len(top) / float(len(res))

            front = filter(lambda x: x[5] == 'front', res)
            front_accuracy = np.sum(np.array(map(lambda x: x[6], front))) / float(len(front))
            n_front = len(front) / float(len(res))

            side = filter(lambda x: x[5] == 'side', res)
            side_accuracy = np.sum(np.array(map(lambda x: x[6], side))) / float(len(side))
            n_side = len(side) / float(len(res))

            row.append([description, accuracy, n_top, top_accuracy, n_front, front_accuracy, n_side, side_accuracy])

        print tabulate(row, headers=['Model & Setup', 'Accuracy', '% Top', 'Top Accuracy', '%Front', 'Front Accuracy', '% Side', 'Side Accuracy'])

# if __name__ == '__main__':
#     # simulator_baseline()
#     # jacquard_baseline()
#     # beam_search_improve()
# #    beam_search_optimize()
# #     input_outputs()
#     # beam_loss()
#     jacquard_vs_cornell_dataset()
#     plt.show()
