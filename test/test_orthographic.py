from unittest import TestCase
from mpl_toolkits.mplot3d import Axes3D
from simulator.simulator import Simulator
from utils.ransac import Plane
from core.orthographic import PointCloud

import pylab as plt
import numpy as np
import os
import h5py
import pptk
import core.orthographic


class TestOrthographic(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOrthographic, self).__init__(*args, **kwargs)
        # self.pcd = self.create_cube()
        self.pcd = np.load('points.npy')

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

    def test_orthographic_projection(self):
        cloud = PointCloud(self.pcd)
        front, side, top, pca = cloud.orthographic_projection()

        ax = self.render_pcd(self.pcd)
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

    def test_to_depth(self):

        cloud = PointCloud(np.load('object_cloud.npy'))

        pos = np.array([0, 0, 0])
        frame = np.eye(3)
        core.orthographic.render_frame(pos, frame[0], frame[1], frame[2], cloud=cloud)

        front, right, top = cloud.orthographic_depth()
        plt.subplot(131)
        plt.imshow(front.img)
        plt.subplot(132)
        plt.imshow(right.img)
        plt.subplot(133)
        plt.imshow(top.img)
        plt.colorbar()
        plt.show()

    def test_inverse_transform(self):
        cloud = PointCloud(np.load('object_cloud.npy'))

        pos = np.array([0, 0, 0])
        frame = np.eye(3)
        core.orthographic.render_frame(pos, frame[0], frame[1], frame[2], cloud=cloud)

        blue = np.array([[0, 0, 1]])
        colors = np.concatenate((np.ones(cloud.cloud.shape), blue))

        front = cloud.front_depth()
        right = cloud.right_depth()
        top = cloud.top_depth()

        points = np.concatenate((cloud.cloud, front.to_object((21, 110)).reshape(1, 3)))
        viewer = pptk.viewer(points)
        viewer.attributes(colors)
        viewer.set(point_size=0.0005)

        points = np.concatenate((cloud.cloud, right.to_object((21, 110)).reshape(1, 3)))
        viewer = pptk.viewer(points)
        viewer.attributes(colors)
        viewer.set(point_size=0.0005)

        points = np.concatenate((cloud.cloud, top.to_object((191, 198)).reshape(1, 3)))
        viewer = pptk.viewer(points)
        viewer.attributes(colors)
        viewer.set(point_size=0.0005)

        plt.subplot(131)
        plt.imshow(front.img)
        plt.scatter([110], [21])
        plt.subplot(132)
        plt.imshow(right.img)
        plt.scatter([110], [21])
        plt.subplot(133)
        plt.imshow(top.img)
        plt.scatter([198], [191])
        plt.show()

    def test_remove_plane(self):
        cloud = PointCloud(self.pcd)
        self.render_pcd(self.pcd)
        cloud.remove_plane()
        self.render_pcd(cloud.cloud)
        plt.show()

    def test_cube(self):
        pc = self.create_cube()
        self.render_pcd(pc)
        plt.show()

    def test_from_file(self):
        cloud = PointCloud.from_file('pcd0116.txt')
        cloud.render()

    def test_filter_roi(self):
        cloud = PointCloud(self.pcd)
        cloud.filter_roi([-0.5, 0.5]*3)
        self.render_pcd(cloud.cloud)
        plt.show()

    def test_filter_and_ransac(self):
        cloud = PointCloud.from_file('pcd0116.txt')
        cloud.filter_roi([0, 1400, -np.inf, np.inf, 0, 500])
        cloud.remove_plane()
        cloud.render()

    def test_full_pipeline(self):
        cloud = PointCloud.from_file('pcd0116.txt')
        # cloud.filter_roi([-np.inf, np.inf, -np.inf, np.inf, 0, 500])
        cloud.filter_roi([0, 1400, -np.inf, np.inf, 0, 500])
        cloud.remove_plane()

        cloud.render()

        cloud.pca()
        plt.subplot(1, 3, 1)
        plt.imshow(cloud.to_depth(index=0))
        plt.subplot(1, 3, 2)
        plt.imshow(cloud.to_depth(index=1))
        plt.subplot(1, 3, 3)
        plt.imshow(cloud.to_depth(index=2))
        plt.show()

        # TODO: finish pipeline

    def test_find_plane(self):
        pcd = np.load('points.npy')
        cloud = PointCloud(pcd)
        mean, normal = cloud.find_plane()

    def test_render(self):
        pcd = np.load('points2.npy')
        cloud = PointCloud(pcd)
        cloud.render()
        cloud.render(use_pptk=False, subsample=0.01)

    # TODO: remove this
    def test_rotate_points_to_sensible_orientation(self):
        cloud = PointCloud.from_npy('points.npy')
        plane_cloud = cloud.find_plane(th=0.02)
        plane_cloud.pca()
        cloud.cloud = plane_cloud.transform(cloud.cloud)
        cloud.cloud[:, 0] += 0.7
        rotated = cloud.rotate([0, 1, 0], np.radians(-45))
        rotated.save('isolated_cloud.npy')

    def test_rotate(self):
        pcd = np.load('isolated_cloud.npy')
        cloud = PointCloud(pcd)
        rotated = cloud.rotate([0, 1, 0], np.radians(45))
        rotated.render()
