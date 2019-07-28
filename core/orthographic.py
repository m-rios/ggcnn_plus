import numpy as np
import pandas as pd
import pptk
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from skimage.filters import gaussian
from scipy.spatial.transform import Rotation as R


class Depth:
    def __init__(self, img, pixel_size, pca):
        self.img = img
        self.pixel_size = pixel_size
        self.pca = pca

        self.fill_missing()

    def blur(self, sigma):
        """Applies gaussian blur"""
        self.img = gaussian(self.img, sigma, preserve_range=True)

    def get_3d_coordinates(self, u, v):
        point = np.reshape([v, u, self.img[u, v]], (1,3))
        return self.pca.inverse_transform(point)

    def fill_missing(self):
        missing_idx = np.isinf(self.img)
        fill_value = np.min(self.img[np.logical_not(missing_idx)])
        self.img[missing_idx] = fill_value


class PointCloud:
    def __init__(self, cloud):
        self.cloud = cloud
        self.pca_ = None
        self._axes = [0, 1, 2]

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

    @classmethod
    def from_npy(cls, fn):
        return cls(np.load(fn))

    @property
    def front(self):
        return self.cloud.T[1:].T

    @property
    def right(self):
        return self.cloud.T[0, 2].T

    @property
    def top(self):
        return self.cloud.T[:2].T

    @property
    def front_depth(self, shape=300):
        return self.to_depth(shape, 0)

    @property
    def right_depth(self, shape=300):
        return self.to_depth(shape, 1)

    @property
    def top_depth(self, shape=300):
        return self.to_depth(shape, 2)

    def pixel_size(self, shape):
        """
        Calculate pixel size in world units
        :param shape: rows or columns of the image (only squared images are supported)
        :return: pixel size
        """
        # Calculate pixel size (same for all axes and all views)
        max_ = np.max(self.cloud, axis=0)
        min_ = np.min(self.cloud, axis=0)

        return np.max(np.abs((max_ - min_)/(shape - 1)))

    def save(self, fn):
        np.save(fn, self.cloud)

    def rotate(self, axis, angle):
        axis = axis/np.linalg.norm(axis)
        r = R.from_rotvec(axis*angle)
        return PointCloud(r.apply(self.cloud))

    def to_depth(self, shape=300, index=0, padding=7, pixel_radius=1):
        """
        Construct depth image from point cloud
        :param shape: output shape of the depth image (only one value, output is squared)
        :param index: column index in pc that defines depth
        :param missing: default value for missing-data
        :return: ndarray of shape shape with depth information
        """
        assert pixel_radius <= padding
        final_shape = shape
        shape -= padding*2
        depth = np.ones((final_shape, final_shape, self.cloud.shape[0])) * -np.inf
        spatial_idx = list({0, 1, 2} - {index})

        # Calculate pixel size (same for all axes and all views)
        minr = np.min(self.cloud[:, spatial_idx[0]])
        minc = np.min(self.cloud[:, spatial_idx[1]])
        maxr = np.max(self.cloud[:, spatial_idx[0]])
        maxc = np.max(self.cloud[:, spatial_idx[1]])

        pixel_size = self.pixel_size(shape)

        for p_idx, p in enumerate(self.cloud):
            r = int((p[spatial_idx[0]] - minr)/pixel_size) + padding
            c = int((p[spatial_idx[1]] - minc)/pixel_size) + padding
            # r = int((p[spatial_idx[0]] - (maxr - minr)/2)/pixel_size) + padding
            # c = int((p[spatial_idx[1]] - (maxc - minc)/2)/pixel_size) + padding
            rs = range(r - pixel_radius, r + pixel_radius + 1)
            cs = range(c - pixel_radius, c + pixel_radius + 1)
            rs, cs = np.meshgrid(rs, cs)
            rs, cs = rs.flatten(), cs.flatten()
            depth[rs, cs, p_idx] = p[index]

        # Positive values occlude negative ones
        depth = np.max(depth, axis=2)

        # Swap rows for columns (e.g. rows vertical but x horizontal)
        depth = depth.T

        # Reverse columns for side view (x points left)
        if index == 1:
            depth = depth[:, ::-1]

        # Reverse rows (origin at upper left corner in image space)
        depth = depth[::-1]

        return Depth(depth, pixel_size, self.pca_)

    def orthographic_projection(self):
        """
        Projects a point cloud into three orthogonal views
        :param pc: point cloud ndarray (N,3) -> N: n features. Contains only the object from which to extract the views
        :return: 3 point clouds corresponding to each orthographic projection
        """
        pca = PCA()
        transformed = pca.fit_transform(self.cloud)
        mu = pca.mean_
        components = pca.components_

        front = transformed.T[1:].T
        side = transformed.T[[0, 2]].T
        top = transformed.T[:2].T

        return front, side, top, pca

    def pca(self, axes=None):
        """
        Uses PCA to center cloud around mean and orient it along its principal axes
        :param axes: axes or features to perform the analysis on. If None all axes are used
        """
        if axes is not None:
            self._axes = axes
        self.pca_ = PCA()
        self.pca_.fit(self.cloud[:, self._axes])
        return PointCloud(self.transform(self.cloud))

    def transform_inverse(self, points):
        """
        Transforms a set of points back to its original space using an existing pca
        """

        if self.pca_ is not None:
            return self.pca_.inverse_transform(points)

        return points

    def transform(self, points):
        if self.pca_ is not None:
            transformed_redux = self.pca_.transform(points[:, self._axes])
            untouched_axes = list(set(range(3)) - set(self._axes))
            return np.insert(transformed_redux, untouched_axes, points[:, untouched_axes], axis=1)
        return points

    def filter_roi(self, roi):
        """
        Removes all points not in roi
        :param roi: [min_x, max_x, min_y, max_y, min_z, max_z]
        """
        assert len(roi) == 6

        mask = np.all(np.logical_and(np.less_equal(self.cloud, roi[1::2]), np.greater_equal(self.cloud, roi[::2])), axis=1)
        return PointCloud(self.cloud[mask, :])

    def find_plane(self, th=10.):
        ransac = RANSACRegressor(residual_threshold=th)
        ransac.fit(self.cloud[:, :2], self.cloud[:, 2])
        inliers = ransac.inlier_mask_
        return PointCloud(self.cloud[inliers])

    def remove_plane(self, th=10.):
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
        return self.remove_points(ransac.inlier_mask_)

    def remove_points(self, mask):
        mask = np.logical_not(mask)
        return PointCloud(self.cloud[mask])

    def render(self, use_pptk=True, subsample=1):
        """
        Renders the cloud
        :param use_pptk: if True will use pptk. Otherwise it will use matplotlib's scatter (slow)
        :param subsample: fraction of the points to render
        """
        selected = np.random.choice(range(self.cloud.shape[0]), int(subsample*self.cloud.shape[0]))
        pcd = self.cloud[selected]
        if use_pptk:
            pptk.viewer(pcd)
        else:
            import pylab as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = Axes3D(fig)

            xs, ys, zs = pcd.T
            ax.scatter(xs, ys, zs, '*', s=0.1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.axis('equal')
            plt.show()

    def plot_views(self, show=True):
        front = self.front_depth
        right = self.right_depth
        top = self.top_depth

        import pylab as plt
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(front.img)
        plt.title('Front')
        plt.subplot(1, 3, 2)
        plt.imshow(right.img)
        plt.title('Right')
        plt.subplot(1, 3, 3)
        plt.imshow(top.img)
        plt.title('Top')
        if show:
            plt.show()


def pipeline():
    original_cloud = PointCloud.from_npy('../test/isolated_cloud.npy')
    plane_cloud = original_cloud.find_plane(th=0.02)
    plane_cloud.pca()
    filtered_cloud = PointCloud(plane_cloud.transform(original_cloud.cloud))
    filtered_cloud = filtered_cloud.filter_roi([-2, 1, -.15, .25, 0, 0.2])
    filtered_cloud = filtered_cloud.remove_plane(th=0.02)
    assert filtered_cloud.cloud.size > 0
    # pre_filtered_cloud = filtered_cloud.remove_plane(th=0.02)
    # while pre_filtered_cloud.cloud.size == 0:
    #     pre_filtered_cloud = filtered_cloud.remove_plane(th=0.02)
    # filtered_cloud = pre_filtered_cloud
    filtered_cloud = filtered_cloud.pca(axes=[0, 1])

    filtered_cloud.plot_views()


if __name__ == '__main__':
    pipeline()
