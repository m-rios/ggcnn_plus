import pylab as plt
import numpy as np

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

    values = np.array(iou_values).astype(np.float)/100.
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
        plot_fn = OUTPUT_PATH+filename+'.eps'
        plt.savefig(plot_fn)
        print 'saved plot as {}'.format(plot_fn)


def simulator_baseline():
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


def jacquard_baseline():
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


def beam_search_improve():
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


def beam_search_optimize():
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

    starting_values = [0.425352112676, 'NaN']
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


def beam_loss():
    epochs, loss_values = _read_beam(RESULTS_PATH + 'beam_search_190926_2111/results.txt')
    epochs_sim, sim_values = _read_simulation(RESULTS_PATH + '190928_1305_beam_search_190926_2111/results.txt')
    plt.subplot(121)
    plt.plot(epochs, loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.plot(epochs_sim, sim_values)
    plt.title('SIM')
    plt.ylabel('% successful grasps')
    plt.xlabel('Epochs')
    plt.suptitle('Loss as heuristic')


if __name__ == '__main__':
    # simulator_baseline()
    # jacquard_baseline()
    # beam_search_improve()
    # beam_search_optimize()
    beam_loss()
    plt.show()
