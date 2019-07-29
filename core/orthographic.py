import numpy as np
import pandas as pd
import pptk
import cv2
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from skimage.filters import gaussian
from scipy.spatial.transform import Rotation as R


class Depth:
    def __init__(self, img, inverse_transform, pixel_radius, index):
        self.img = img
        self.inverse_transform = inverse_transform
        self.index = index

        self._apply_radius(pixel_radius)
        self.fill_missing()

    def _apply_radius(self, pixel_radius):
        filled = np.argwhere(np.logical_not(np.isinf(self.img)))
        new_img = np.ones(self.img.shape) * -np.inf

        for r, c in filled:
            rs = range(r - pixel_radius, r + pixel_radius + 1)
            cs = range(c - pixel_radius, c + pixel_radius + 1)
            new_img[np.ix_(rs, cs)] = np.maximum(new_img[np.ix_(rs, cs)], np.full((2*pixel_radius + 1,)*2, self.img[r, c]))

        self.img = new_img

    def blur(self, sigma):
        """Applies gaussian blur"""
        self.img = gaussian(self.img, sigma, preserve_range=True)

    def fill_missing(self):
        missing_idx = np.isinf(self.img)
        fill_value = np.min(self.img[np.logical_not(missing_idx)])
        self.img[missing_idx] = fill_value

    def to_object(self, uv):
        uv = np.array(uv)
        u_ = uv[1]
        v_ = self.img.shape[0] - uv[0] - 1
        xy = self.inverse_transform(np.array([u_, v_]))
        d = self.img[uv[0], uv[1]]
        return np.insert(xy, self.index, d)


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
        depth = np.ones((final_shape, final_shape)) * -np.inf
        spatial_idx = list({0, 1, 2} - {index})

        # Calculate pixel size (same for all axes and all views)
        min_ = np.min(self.cloud[:, spatial_idx], axis=0)
        max_ = np.max(self.cloud[:, spatial_idx], axis=0)

        pixel_size = self.pixel_size(shape)
        center = (max_ - min_)/2.  # Center of the object in world units
        t = (shape/2. - 1 - center/pixel_size).astype(np.int)  # Translation from center w.r.t. min_ to image center

        for p in self.cloud:
            r, c = ((p[spatial_idx] - min_)/pixel_size).astype(np.int) + padding + t
            depth[r, c] = max(p[index], depth[r, c])

        # Swap rows for columns (e.g. rows vertical but x horizontal)
        depth = depth.T

        # Reverse columns for side view (x points left)
        if index == 1:
            depth = depth[:, ::-1]

        # Reverse rows (origin at upper left corner in image space)
        depth = depth[::-1]

        def inverse_transform(pixel):
            return (pixel - padding - t)*pixel_size + min_

        return Depth(depth, inverse_transform, pixel_radius, index)

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
        # if self.pca_ is not None:
        #     return self.pca_.inverse_transform(points)
        #
        # return points

        if self.pca_ is not None:
            transformed_redux = self.pca_.inverse_transform(points[:, self._axes])
            untouched_axes = list(set(range(3)) - set(self._axes))
            return np.insert(transformed_redux, untouched_axes, points[:, untouched_axes], axis=1)
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

    def find_plane(self, th=.02):
        ransac = RANSACRegressor(residual_threshold=th)
        ransac.fit(self.cloud[:, :2], self.cloud[:, 2])
        inliers = ransac.inlier_mask_
        return PointCloud(self.cloud[inliers])

    def remove_plane(self, th=.02):
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

    def render(self):
        """
        Renders the cloud
        :param subsample: fraction of the points to render
        """
        return pptk.viewer(self.cloud)

    def plot(self, subsample=1):
        selected = np.random.choice(range(self.cloud.shape[0]), int(subsample*self.cloud.shape[0]))
        pcd = self.cloud[selected]
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
            print 'Done plotting'
            plt.show()


class OrthoNet:
    def __init__(self):
        self.cloud = None

    def predict(self, cloud, predictor, roi=None):
        if roi is None:
            roi = [-np.inf, np.inf] * 3
        self.cloud = PointCloud(cloud)
        plane_cloud = self.cloud.find_plane()
        plane_cloud.pca()
        wrt_table_cloud = PointCloud(plane_cloud.transform(self.cloud.cloud))
        wrt_table_cloud = wrt_table_cloud.filter_roi(roi)
        wrt_table_cloud = wrt_table_cloud.remove_plane()
        assert wrt_table_cloud.cloud.size > 0
        # wo_table = wrt_table_cloud.remove_plane()
        # while wo_table.cloud.size == 0:
        #     print 'Plane removal was not successful, trying again'
        #     wo_table = wrt_table_cloud.remove_plane()
        # wrt_table_cloud = wo_table
        global wrt_object_cloud
        wrt_object_cloud = wrt_table_cloud.pca(axes=[0, 1])

        depth = wrt_object_cloud.top_depth
        point, orientation = predictor(depth)

        point_wrt_table = wrt_object_cloud.transform_inverse(point)
        point_wrt_world = plane_cloud.transform_inverse(point_wrt_table)

        return point_wrt_world




    def predict_debug(self, cloud, roi=None):
        if roi is None:
            roi = [-np.inf, np.inf] * 3
        self.cloud = PointCloud(cloud)
        plane_cloud = self.cloud.find_plane()
        plane_cloud.pca()
        wrt_table_cloud = PointCloud(plane_cloud.transform(self.cloud.cloud))
        wrt_table_cloud = wrt_table_cloud.filter_roi(roi)
        wo_table = wrt_table_cloud.remove_plane()
        while wo_table.cloud.size == 0:
            print 'Plane removal was not successful, trying again'
            wo_table = wrt_table_cloud.remove_plane()
        wrt_table_cloud = wo_table
        global wrt_object_cloud
        wrt_object_cloud = wrt_table_cloud.pca(axes=[0, 1])

        # front = wrt_object_cloud.front_depth
        global right
        right = wrt_object_cloud.top_depth
        # top = wrt_object_cloud.top_depth

        global last_lmb_event
        last_lmb_event = cv2.EVENT_LBUTTONUP

        global center
        center = (0, 0)

        def handle_mouse(event, x, y, flags, param):
            global last_lmb_event
            global center
            global color
            global right
            global end
            global start
            global wrt_object_cloud

            # Mouse click
            if event == cv2.EVENT_LBUTTONDOWN:
                print 'Mouse clicked at {}'.format((x, y))
                last_lmb_event = event
                center = (x, y)
            # Mouse release or drag
            elif event == cv2.EVENT_LBUTTONUP:
                print 'Mouse released at {}'.format((x, y))
                last_lmb_event = event
                # depth = right.img[center[::-1]]
                # print depth
                cloud = wrt_object_cloud.cloud
                point = right.to_object(center[::-1])
                points = np.concatenate((cloud, np.reshape(point, (1, 3))))
                viewer = pptk.viewer(points)
                viewer.attributes(np.concatenate((np.ones(cloud.shape), [[1, 0, 0]])))
                viewer.set(point_size=0.001)
            elif event == cv2.EVENT_MOUSEMOVE and last_lmb_event == cv2.EVENT_LBUTTONDOWN:
                end = (x, y)
                start = tuple(np.clip(np.subtract(np.multiply(center, 2), end), (0, 0), (299, 299)).astype(np.int))
                to_draw = np.copy(color)
                cv2.line(to_draw, start, end, (0, 255, 0), 2)
                cv2.imshow('views', to_draw)

        cv2.namedWindow('views')
        cv2.setMouseCallback('views', handle_mouse)
        normalized = (255 * (right.img - np.min(right.img))/(np.max(right.img) - np.min(right.img))).astype(np.uint8)
        global color
        color = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        cv2.imshow('views', color)
        print 'image rendered'
        while True:
            key_press = cv2.waitKey()
            if key_press is 113:
                break


def manual_predictor(depth_img):
    global last_lmb_event
    last_lmb_event = cv2.EVENT_LBUTTONUP

    normalized = (255 * (depth_img.img - np.min(depth_img.img))/(np.max(depth_img.img) - np.min(depth_img.img))).astype(np.uint8)
    global color
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
    cv2.imshow('views', color)

    global point
    point = None

    def handle_mouse(event, x, y, flags, param):
        global last_lmb_event
        global center
        global color
        global end
        global start
        global point

        # Mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            last_lmb_event = event
            center = (x, y)
        # Mouse release or drag
        elif event == cv2.EVENT_LBUTTONUP:
            last_lmb_event = event
            point = param.to_object(center[::-1])
        elif event == cv2.EVENT_MOUSEMOVE and last_lmb_event == cv2.EVENT_LBUTTONDOWN:
            end = (x, y)
            start = tuple(np.clip(np.subtract(np.multiply(center, 2), end), (0, 0), (299, 299)).astype(np.int))
            to_draw = np.copy(color)
            cv2.line(to_draw, start, end, (0, 255, 0), 2)
            cv2.imshow('views', to_draw)

    cv2.namedWindow('views')
    cv2.setMouseCallback('views', handle_mouse, depth_img)
    cv2.imshow('views', color)
    while point is None:
        cv2.waitKey(1)
    return np.reshape(point, (1, 3)), None

def ros():
    import rospy
    from visualization_msgs.msg import Marker
    import ros_numpy
    from sensor_msgs.msg import PointCloud2
    rospy.init_node('orthonet')

    pub = rospy.Publisher('marker', Marker, latch=True, queue_size=1)
    cloud_data = rospy.wait_for_message('/camera/depth/points', PointCloud2)
    xyzs = ros_numpy.point_cloud2.get_xyz_points(ros_numpy.numpify(cloud_data))
    rospy.sleep(1)

    onet = OrthoNet()
    point = onet.predict(xyzs, roi=[-2, 1, -.15, .25, 0, 0.2], predictor=manual_predictor)

    marker = Marker()

    marker.header.frame_id = "camera_depth_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = 'debug'
    marker.id = 1
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.pose.orientation.w = 1
    marker.pose.position.x = point[0, 0]
    marker.pose.position.y = point[0, 1]
    marker.pose.position.z = point[0, 2]
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.b = 1.0

    # print 'sending'
    # while not rospy.is_shutdown():
    pub.publish(marker)
    rospy.sleep(1)

if __name__ == '__main__':
    ros()
    # cloud = PointCloud.from_npy('../test/isolated_cloud.npy')
    # onet = OrthoNet()
    # # onet.predict_debug(cloud, roi=[-2, 1, -.15, .25, 0, 0.2])
    # point = onet.predict(cloud, manual_predictor,roi=[-2, 1, -.15, .25, 0, 0.2])
    # points = np.concatenate((cloud.cloud, point))
    # viewer = pptk.viewer(points)
    # viewer.attributes(np.concatenate((np.ones(cloud.cloud.shape), [[0, 0, 1]])))
    # viewer.set(point_size=0.001)
