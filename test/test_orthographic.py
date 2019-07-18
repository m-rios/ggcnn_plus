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
        # sim = Simulator(gui=False, use_egl=False)
        #
        # # Load random scene
        # scenes_ds = h5py.File('../data/scenes/shapenet_1.hdf5', 'r')
        # scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        # sim.restore(scene, os.environ['MODELS_PATH'])
        #
        # # Get pcd
        # self.pcd = sim.cam.point_cloud()

    def create_cube(self):
        base = Plane.from_point_vector([0, 0, 0], [0, 0, 1])
        top = Plane.from_point_vector([0, 0, 1], [0, 0, 1])
        left = Plane.from_point_vector([0, 0, 0], [1, 0, 0])
        right = Plane.from_point_vector([1, 0, 0], [1, 0, 0])

        n = 200
        base_pc = base.sample(n=n, noise=0)
        top_pc = top.sample(n=n, noise=0)
        left_pc = left.sample(n=n, noise=0)
        right_pc = right.sample(n=n, noise=0)

        pc = np.concatenate((base_pc, top_pc, left_pc, right_pc))
        return pc

    def render_frame(self, ax, mean, components):
        xs, ys, zs = (np.vstack((mean, components[0]))).T
        ax.plot(xs, ys, zs, color='red', linewidth=2)
        xs, ys, zs = (np.vstack((mean, components[1]))).T
        ax.plot(xs, ys, zs, color='green', linewidth=2)
        xs, ys, zs = (np.vstack((mean, components[2]))).T
        ax.plot(xs, ys, zs, color='blue', linewidth=2)

        return ax

    def render_pcd(self, pcd):
        fig = plt.figure()
        ax = Axes3D(fig)

        xs, ys, zs = pcd.T
        ax.scatter(xs, ys, zs)
        print 'WARNING: Don\'t forget to call plt.show()'
        return ax

    def test_plot_orthogonal_components(self):
        cube = self.create_cube()
        ortho.extract_ortho_views(cube)
        ax = self.render_pcd(cube)
        ax.plot([0, 1], [0, 0], [0, 0], 'r')
        plt.show()


    def test_extract_ortho_views(self):
        cube = self.create_cube()
        front, side, top, pca = ortho.extract_ortho_views(cube)

        ax = self.render_pcd(cube)
        ax = self.render_frame(ax, pca.mean_, pca.components_)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.scatter(front[:, 0], front[:, 1])
        plt.title('Front')
        plt.subplot(1, 3, 2)
        plt.scatter(side[:, 0], side[:, 1])
        plt.title('Side')
        plt.subplot(1, 3, 3)
        plt.scatter(top[:, 0], top[:, 1])
        plt.title('Top')
        plt.show()

    def test_depth_from_pcd(self):
        cube = self.create_cube()

        ax = self.render_pcd(cube)
        self.render_frame(ax, np.zeros(3), np.eye(3))

        plt.figure()
        plt.subplot(2, 2, 1)
        xs, ys = cube[:, [1, 2]].T
        plt.scatter(xs, ys)
        plt.title('Front')
        plt.subplot(2, 2, 2)
        xs, ys = cube[:, [0, 2]].T
        plt.scatter(-xs, ys)
        plt.title('Right')
        plt.subplot(2, 2, 3)
        xs, ys = cube[:, [0, 1]].T
        plt.scatter(xs, ys)
        plt.title('Top')

        front = ortho.depth_from_pc(cube, index=0)
        right = ortho.depth_from_pc(cube, index=1)
        top = ortho.depth_from_pc(cube, index=2)

        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(front)
        plt.title('Front')
        plt.subplot(2, 2, 2)
        plt.imshow(right)
        plt.title('Right')
        plt.subplot(2, 2, 3)
        plt.imshow(top)
        plt.title('Top')

        plt.show()

    def test_remove_plane(self):
        self.render_pcd(self.pcd)
        self.render_pcd(ortho.remove_plane(self.pcd))
        plt.show()

    def test_cube(self):
        pc = self.create_cube()
        self.render_pcd(pc)
        plt.show()
