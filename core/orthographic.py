import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from utils.ransac import ransac


def extract_ortho_views(pc):
    """
    Projects a point cloud into three orthogonal views
    :param pc: point cloud ndarray (N,3) -> N: n features. Contains only the object from which to extract the views
    :return: 3 depth images corresponding to each orthographic projection
    """
    pca = PCA()
    analysis = pca.fit(pc)
    mu = analysis.mean_
    components = analysis.components_


def remove_plane(pc, k=10, epsilon=0.005):
    """
    Removes the largest plane in the point cloud (assumed to be the workspace surface)
    :param pc: ndarray (N,3) representing the point cloud
    :param k: number of iterations for plane segmentation
    :param epsilon: max error to consider point lies within model (ransac)
    :return: a pc without the largest plane
    """
    _, _, plane_idxs = ransac(pc, k=k, epsilon=epsilon)
    object_idxs = list(set(range(pc.shape[0])) - set(plane_idxs))
    object_pc = pc[object_idxs]

    return object_pc


if __name__ == '__main__':
    depth = np.load('../test/depth_inpainted.npy')
    rows, cols, depth = depth_to_pcd(depth)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(rows, cols, depth)
    plt.show()
