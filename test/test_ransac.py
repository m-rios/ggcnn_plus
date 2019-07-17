from unittest import TestCase
from mpl_toolkits.mplot3d import Axes3D

import utils.ransac
import pylab as plt
import numpy as np


class TestRansac(TestCase):
    def test_ransac(self):
        plane = utils.ransac.Plane.random_plane(max_D=0.1)
        plane_points = plane.sample(n=100)
        noise_points = np.random.uniform(-1, 1, (200, 3))
        points = np.append(plane_points, noise_points, axis=0)

        _, _, idxs = utils.ransac.ransac(points)

        fig = plt.figure()
        ax = Axes3D(fig)
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.scatter(points[idxs, 0], points[idxs, 1], points[idxs, 2], 'r')
        plt.show()

