from unittest import TestCase
from mpl_toolkits.mplot3d import Axes3D
from simulator.simulator import Simulator
from utils.ransac import Plane

import core.orthographic as ortho
import pylab as plt
import numpy as np
import os
import h5py


class TestOrthographic(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOrthographic, self).__init__(*args, **kwargs)
        sim = Simulator(gui=False, use_egl=False)

        # Load random scene
        scenes_ds = h5py.File('../data/scenes/shapenet_1.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        # Get pcd
        self.pcd = sim.cam.point_cloud()

    def render_pcd(self, pcd):
        fig = plt.figure()
        ax = Axes3D(fig)

        xs, ys, zs = pcd.T
        ax.scatter(xs, ys, zs)
        print 'WARNING: Don\'t forget to call plt.show()'

    def test_extract_ortho_views(self):
        ortho.extract_ortho_views(self.pcd)

    def test_remove_plane(self):
        self.render_pcd(self.pcd)
        self.render_pcd(ortho.remove_plane(self.pcd))
        plt.show()

    def test_cube(self):
        base = Plane.from_point_vector([0, 0, 0], [0, 0, 1])
        top = Plane.from_point_vector([0, 0, 1], [0, 0, 1])
        left = Plane.from_point_vector([0, 0, 0], [1, 0, 0])
        right = Plane.from_point_vector([1, 0, 0], [1, 0, 0])

        base_pc = base.sample(n=100, noise=0)
        top_pc = top.sample(n=100, noise=0)
        left_pc = left.sample(n=100, noise=0)
        right_pc = right.sample(n=100, noise=0)

        pc = np.concatenate((base_pc, top_pc, left_pc, right_pc))
        self.render_pcd(pc)
        plt.show()
