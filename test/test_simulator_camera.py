from unittest import TestCase

from keras.backend import epsilon

from simulator.simulator import Simulator
from mpl_toolkits.mplot3d import Axes3D
from utils.ransac import ransac

import h5py
import numpy as np
import os
import pylab as plt
import pptk


class TestSimulatorCamera(TestCase):
    def test_point_cloud(self):
        sim = Simulator(gui=False, use_egl=False)

        # Load random scene
        scenes_ds = h5py.File('../data/scenes/shapenetsem40_5.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        # Get pcd
        pcd = sim.cam.point_cloud()
        self.assertTrue(pcd.shape[0] > 0)
        pptk.viewer(pcd)

    def test_plane_segmentation(self):
        sim = Simulator(gui=False, use_egl=False)

        # Load random scene
        scenes_ds = h5py.File('../data/scenes/shapenet_1.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        # Get pcd
        pcd = sim.cam.point_cloud()
        _, _, plane_idx = ransac(pcd, k=10, epsilon=0.005)
        not_plane_idx = list(set(range(pcd.shape[0])) - set(plane_idx))

        fig = plt.figure()
        ax = Axes3D(fig)
        xs, ys, zs = pcd[plane_idx].T
        ax.scatter(xs, ys, zs)
        xs, ys, zs = pcd[not_plane_idx].T
        ax.scatter(xs, ys, zs, 'r')
        plt.show()
