import numpy as np
import pandas as pd
import pptk
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor


class PointCloud:
    def __init__(self, cloud):
        self.cloud = cloud

    def __getitem__(self, item):
        return self.cloud[item]

    def __setitem__(self, key, value):
        self.cloud[key] = value

    @classmethod
    def from_file(cls, fn):
        # If this breaks check that 'DATA ascii' is in line 9 in the file
        df = pd.read_csv(fn, skiprows=range(10), names=['x', 'y', 'z', 'rgb', 'index'], delimiter=' ')
        cloud = df[['x', 'y', 'z']].to_numpy()
        return cls(cloud)

    def to_depth(self, shape=300, index=0, missing=-1):
        """
        Construct depth image from point cloud
        :param shape: output shape of the depth image (only one value, output is squared)
        :param index: column index in pc that defines depth
        :param missing: default value for missing-data
        :return: ndarray of shape shape with depth information
        """
        shape = (shape, shape)
        depth = np.ones(shape) * missing
        spatial_idx = list({0, 1, 2} - {index})

        # row and column increments in world units
        minr = np.min(self.cloud[:, spatial_idx[0]])
        maxr = np.max(self.cloud[:, spatial_idx[0]])
        minc = np.min(self.cloud[:, spatial_idx[1]])
        maxc = np.max(self.cloud[:, spatial_idx[1]])
        d = max(abs(maxr - minr)/(shape[0] - 1), abs(maxc - minc)/(shape[1] - 1))  # same increment for both axes

        for p in self.cloud:
            r = int((p[spatial_idx[0]] - minr)/d)
            c = int((p[spatial_idx[1]] - minc)/d)
            depth[r, c] = p[index] if depth[r, c] == missing else max(depth[r, c], p[index])

        # Swap rows for columns (e.g. rows vertical but x horizontal)
        depth = depth.T

        # Reverse columns for side view (x points left)
        if index == 1:
            depth = depth[:, ::-1]

        # Reverse rows (origin at upper left corner in image space)
        depth = depth[::-1]

        return depth

    def orthographic_projection(self):
        """
        Projects a point cloud into three orthogonal views
        :param pc: point cloud ndarray (N,3) -> N: n features. Contains only the object from which to extract the views
        :return: 3 depth images corresponding to each orthographic projection
        """
        pca = PCA()
        transformed = pca.fit_transform(self.cloud)
        mu = pca.mean_
        components = pca.components_

        front = transformed.T[1:].T
        side = transformed.T[[0, 2]].T
        top = transformed.T[:2].T

        return front, side, top, pca

    def filter_roi(self, roi):
        """
        Removes all points not in roi
        :param roi: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        assert len(roi) == 6

        mask = np.all(np.logical_and(np.less_equal(self.cloud, roi[1::2]), np.greater_equal(self.cloud, roi[::2])), axis=1)
        self.cloud = self.cloud[mask, :]

    def remove_plane(self, th=10):
        """
        Removes the largest plane in the point cloud (assumed to be the workspace surface)
        :param pc: ndarray (N,3) representing the point cloud
        :param n: minimum number of points to estimate a model
        :param k: max number of iterations
        :param epsilon: threshold to accept a data point as part of the model
        :param d: fraction of data points needed to lie within model for model for it to be accepted as valid
        :return: a pc without the largest plane
        """
        ransac = RANSACRegressor(residual_threshold=th)
        ransac.fit(self.cloud[:, :2], self.cloud[:, 2])
        mask = np.logical_not(ransac.inlier_mask_)
        self.cloud = self.cloud[mask, :]

    def render(self):
        pptk.viewer(self.cloud)
