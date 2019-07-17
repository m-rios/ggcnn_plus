import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

# pcd = np.random.rand(15, 3)
# tree = KDTree(pcd, leaf_size=2)
#
# dist, ind = tree.query(pcd[:1], k=3)


class Plane:
    def __init__(self, normal, D):
        self.n = np.array(normal)
        self.D = D

        self.v = np.random.rand(3)
        self.v -= self.v.dot(self.n) * self.n
        self.w = np.cross(self.n, self.v)
        try:
            self.mu = [0, 0, self.D/self.n[2]]
        except:
            try:
                self.mu = [0, self.D/self.n[1] , 0]
            except:
                self.mu = [self.D/self.n[0], 0, 0]

    @classmethod
    def random_plane(cls, max_D=10):
        normal = np.random.rand(3)
        normal = normal / np.linalg.norm(normal)
        D = np.random.rand() * max_D
        return cls(normal, D)

    @classmethod
    def from_point_vector(cls, point, vector):
        D = np.dot(point, vector)
        return cls(vector, D)

    def parametric(self, r, s):
        return r*self.v + s*self.w + self.mu

    def sample(self, n=50, noise=0.1, limits=(-1, 1)):
        """
        Generates a random point cloud that resembles a plane
        :return:
        """
        parameters = np.random.uniform(limits[0], limits[1], (n, 2)) # Uniform distribution of samples on plane coordinates
        samples = np.zeros((n, 3))
        for p in range(parameters.shape[0]):
            samples[p] = self.parametric(parameters[p][0], parameters[p][1])

        noise = np.random.rand(n, 3) * noise
        samples += noise

        return samples

    def distance(self, point):
        """
        Computes the distance between a point and the plane
        :param point: 3dim array representing the point
        :return: scalar value with L2 distance
        """
        point = np.array(point)
        assert point.shape == (3,), 'Expected point to be of shape (3,) but "{}" found instead'.format(point)
        qp = point - self.mu
        return np.abs(np.dot(qp, self.n))

    @classmethod
    def fit(cls, points):
        """
        Fits a plane to a point cloud using PCA
        :param points: np.array of shape (N, 3) where N is the number of points in the cloud
        :return: a plane object fitted to the given data
        """
        pca = PCA()
        analysis = pca.fit(points)
        normal = analysis.components_[2]
        mu = analysis.mean_
        return cls.from_point_vector(mu, normal), analysis.explained_variance_[2]


# def ransac(points, k=10, epsilon=0.1, d=0.005):
#     """
#     Inspired by https://stackoverflow.com/questions/28731442/detecting-set-of-planes-from-point-cloud
#     :param points: np array rows = n samples cols = n dimensions of sample (3D -> 3)
#     :param k: max number of iterations
#     :param epsilon: threshold to accept a data point as part of the model
#     :param d: fraction of data points needed to lie within model for model for it to be accepted as valid
#     :return: the index in points of the points that lie in the largest plane
#     """
#     points = np.array(points)
#     tree = KDTree(points, leaf_size=2)
#
#     # pca = PCA()
#     best_err = np.inf
#     best_model = None
#     best_subset = np.empty((0, 3))
#
#     for it in range(k):
#         seed_point = points[np.random.randint(points.shape[0])].reshape(1, -1)
#         maybe_inliers_idx = tree.query(seed_point, k=20)[1].squeeze()
#         maybe_inliers = points[maybe_inliers_idx]
#         outliers_idx = list(set(range(points.shape[0])) - set(maybe_inliers_idx))
#
#         maybe_model, _ = Plane.fit(maybe_inliers)
#
#         also_inliers_idx = np.where(np.linalg.norm(points[outliers_idx], axis=1) < epsilon)[0]
#         also_inliers = points[also_inliers_idx]
#
#         if also_inliers.shape[0] > points.shape[0] * d:
#             total_inliers = np.append(maybe_inliers, also_inliers, axis=0)
#             better_model, variance = Plane.fit(total_inliers)
#             if variance < best_err:
#                 best_model = better_model
#                 best_err = variance
#                 best_subset = total_inliers
#
#     return best_model, best_err, best_subset

def ransac(points, n=20, k=100, epsilon=0.1, d=0.05):
    """
    Inspired by https://stackoverflow.com/questions/28731442/detecting-set-of-planes-from-point-cloud
    :param points: np array rows = n samples cols = n dimensions of sample (3D -> 3)
    :param n: minimum number of points to estimate a model
    :param k: max number of iterations
    :param epsilon: threshold to accept a data point as part of the model
    :param d: fraction of data points needed to lie within model for model for it to be accepted as valid
    :return: the index in points of the points that lie in the largest plane
    """
    best_fit = None
    best_err = np.inf
    best_subset = None

    tree = KDTree(points)

    for _ in range(k):
        maybe_inliers_idx = tree.query(points[np.random.randint(points.shape[0])].reshape(1, -1), k=n)[1].squeeze()
        maybe_model, _ = Plane.fit(points[maybe_inliers_idx])

        candidates_idx = list(set(range(points.shape[0])) - set(maybe_inliers_idx))
        also_inliers_idx = []
        for candidate_idx in candidates_idx:
            if maybe_model.distance(points[candidate_idx]) < epsilon:
                also_inliers_idx.append(candidate_idx)

        if len(also_inliers_idx) > points.shape[0] * d:
            total_inliers = np.append(maybe_inliers_idx, also_inliers_idx)
            better_model, variance = Plane.fit(points[total_inliers])
            if variance < best_err:
                best_fit = better_model
                best_err = variance
                best_subset = total_inliers

    return best_fit, best_err, best_subset


if __name__ == '__main__':
    # plane = Plane.random_plane()
    plane = Plane.from_point_vector([0, 0, 0], [0, 1, 1])
    samples = plane.sample(800, 0.01)
    ransac(samples)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, 0.5)
    ax.set_zlim(-1, 1)
    # ax.set_aspect('equal')
    plt.show()

