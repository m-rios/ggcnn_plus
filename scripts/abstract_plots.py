import final_plots as fp
import pylab as plt
import numpy as np


def beam_search_optimize():
    fp.linewidth = 1.6
    plt.rc('axes', linewidth=1.3)
    plt.figure(figsize=(7, 2.5))
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
    colors = [fp.red, 'k']
    marker = '.'

    for idx, iou_path in enumerate(iou_paths):
        iou_results_fn = iou_path + 'results.txt'

        epochs, iou_values = fp._read_beam(iou_results_fn)

        if idx == 0:
            epochs = np.delete(epochs, -1)
            iou_values = np.delete(iou_values, -1)

        plt.plot(epochs, iou_values, color=colors[idx], marker=marker, linewidth=fp.linewidth, markersize=7)

    for idx, sim_path in enumerate(sim_paths):
        sim_results_fn = sim_path + 'results.txt'

        sim_epochs, sim_values = fp._read_simulation(sim_results_fn)
        sim_epochs = np.insert(sim_epochs, 0, -1) + 1
        sim_values = np.insert(sim_values, 0, starting_values[idx])

        if idx == 0:
            sim_epochs = np.delete(sim_epochs, -1)
            sim_values = np.delete(sim_values, -1)

        plt.plot(sim_epochs, sim_values, color=colors[idx], marker=marker, dashes=fp.dash, linewidth=fp.linewidth, markersize=7)

    fontsize = 14
    ax = plt.gca()

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    plt.legend(['narrow IOU > 0.25', 'shallow IOU > 0.25', 'narrow SIM', 'shallow SIM'], markerscale=0)
    plt.xlabel('Depth', fontsize=12, labelpad=-7)
    plt.ylabel('Successful Grasp Rate', fontsize=12)
    plt.grid(True)
    plt.subplots_adjust(bottom=0.15)
    fp._save_plot('beam_search_optimize')


def beam_search_improve():
    fp.linewidth = 1.6
    plt.rc('axes', linewidth=1.3)
    plt.figure(figsize=(7, 5))
    iou_paths = [
        '/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/',
        '/Users/mario/Developer/msc-thesis/data/networks/beam_search_2+8/',
        '/Users/mario/Developer/msc-thesis/data/networks/beam_search_transpose/',
        '/Users/mario/Developer/msc-thesis/data/networks/beam_search_nl_2/',
    ]

    sim_paths = [
        # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_2/',
        # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_5/',
        # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_10/',
        # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_2+8/',
        # '/Users/mario/Developer/msc-thesis/data/results/190922_1959_beam_search_last/'
        '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_2/',
        '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_2+8/',
        '/Users/mario/Developer/msc-thesis/data/results/190926_1229_beam_search_last/',
        '/Users/mario/Developer/msc-thesis/data/results/191107_1007_beam_search_nl_2/'
    ]

    # colors = [red, 'g', 'b', 'k']
    markers = ['.'] * len(iou_paths)

    plt.subplot(211)
    for idx, iou_path in enumerate(iou_paths):
        iou_results_fn = iou_path + 'results.txt'

        epochs, iou_values = fp._read_beam(iou_results_fn)
        plt.plot(epochs, iou_values, marker=markers[idx], linewidth=fp.linewidth, markersize=7)

    fontsize = 14
    ax = plt.gca()

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    plt.title('IOU > 0.25', fontweight='bold')
    plt.xlabel('Depth', fontsize=12, labelpad=-7)
    plt.ylabel('Successful Grasp Rate', fontsize=12)
    plt.legend(['e=2', 'e=2 r = 8', 'e=2 transpose', 'e=2 no lookahead'], markerscale=0)
    plt.grid('on')

    plt.subplot(212)
    for idx, sim_path in enumerate(sim_paths):
        sim_results_fn = sim_path + 'results.txt'

        sim_epochs, sim_values = fp._read_simulation(sim_results_fn)
        sim_epochs = np.insert(sim_epochs, 0, -1) + 1
        sim_values = np.insert(sim_values, 0, fp.vanilla_ggcnn)

        plt.plot(sim_epochs, sim_values, marker=markers[idx], dashes=fp.dash, linewidth=fp.linewidth, markersize=7)

    fontsize = 14
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    plt.title('SIM', fontweight='bold')
    plt.xlabel('Depth', fontsize=12, labelpad=-7)
    plt.ylabel('Successful Grasp Rate', fontsize=12)
    plt.legend(['e=2', 'e=2 r = 8', 'e=2 transpose', 'e=2 no lookahead'], markerscale=0)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.grid(True)

    fp._save_plot('beam_search_improve')


def ggcnn_demo():
    import h5py
    from core.network import Network
    from skimage.filters import gaussian
    from ggcnn.dataset_processing.grasp import detect_grasps
    from ggcnn.dataset_processing.grasp import BoundingBoxes, BoundingBox
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ds = h5py.File('/Users/mario/Developer/msc-thesis/data/scenes/shapenetsem40_5.hdf5', 'r')
    depth = ds['depth'][31]
    net = Network('/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5')
    pos, ang, wid = net.predict(depth)

    pos = gaussian(pos.squeeze(), 5.0, preserve_range=True)
    ang = ang.squeeze()
    wid = gaussian(wid.squeeze(), 5.0, preserve_range=True)
    gs = detect_grasps(pos, ang, width_img=wid, no_grasps=1)

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(221)
    im = plt.imshow(pos)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[np.min(pos[:]), np.max(pos[:])], orientation='horizontal')
    cbar.ax.set_xticklabels(['0', '0.7'])

    ax = plt.subplot(222)
    im = plt.imshow(ang)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[np.min(ang[:]), 0, np.max(ang[:])], orientation='horizontal')
    cbar.ax.set_xticklabels([-1, 0, 1])

    ax = plt.subplot(223)
    im = plt.imshow(wid)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, ticks=[np.min(wid[:]), np.max(wid[:])], orientation='horizontal')
    cbar.ax.set_xticklabels(['0', '74'])

    ax = plt.subplot(224)
    plt.imshow(depth)
    gs[0].plot(ax, 'r')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.045, hspace=0.1, wspace=-0.1)

    # plt.tight_layout()

def table():
    from keras.models import load_model
    original = load_model('/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')
    print 'Keras {}'.format(original.count_params())
    extended = load_model('/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5')
    print 'Keras {}'.format(extended.count_params())
    optimized = load_model('/Users/mario/Developer/msc-thesis/data/networks/beam_search_narrow/arch_C9x9x8_C9x9x8_C5x5x4_C3x3x2_T3x3x2_T5x5x4_T5x5x4_T5x5x4_T5x5x4_T9x9x8_T9x9x8_depth_4_model.hdf5')
    print 'Keras {}'.format(optimized.count_params())

if __name__ == '__main__':
    beam_search_optimize()
    plt.show()
