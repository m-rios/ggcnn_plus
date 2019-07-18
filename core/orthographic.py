import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from utils.ransac import ransac


class PointCloud:
    def __init__(self, cloud):
        self.cloud = cloud

    @classmethod
    def from_file(cls, fn):
        cloud = np.empty((1, 3))
        with open(fn, 'r') as f:
            for line in f.readlines():

                fields = line.split()

                if len(fields) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(fields[0])
                except ValueError:
                    continue

                point = np.reshape(fields[:3], (1, 3))
                cloud = np.append(cloud, point)
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


    def remove_plane(self, k=10, epsilon=0.005):
        """
        Removes the largest plane in the point cloud (assumed to be the workspace surface)
        :param pc: ndarray (N,3) representing the point cloud
        :param k: number of iterations for plane segmentation
        :param epsilon: max error to consider point lies within model (ransac)
        :return: a pc without the largest plane
        """
        _, _, plane_idxs = ransac(self.cloud, k=k, epsilon=epsilon)
        object_idxs = list(set(range(self.cloud.shape[0])) - set(plane_idxs))
        object_pc = self.cloud[object_idxs]

        return object_pc


if __name__ == '__main__':
    depth = np.load('../test/depth_inpainted.npy')
    rows, cols, depth = depth_to_pcd(depth)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(rows, cols, depth)
    plt.show()
