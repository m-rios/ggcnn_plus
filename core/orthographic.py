import numpy as np
import pandas as pd
import pptk
import cv2
import time
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from skimage.filters import gaussian
from scipy.spatial.transform import Rotation as R


class Transform:
    def __init__(self, pca):
        self.pca = pca

    def transform(self, cloud, axes=None):
        """
        Transforms from existing space to pca space
        :param cloud: ndarray (N, 3) with the points to be transformed or PointCloud object
        :return: ndarray(N, 3) with the transformed points
        """
        axes = axes or range(3)
        original_shape = None
        assert len(axes) == self.pca.n_components_, 'Expected len of axes to be \'{}\', found \'{}\' instead'.format(self.pca.n_components_, len(axes))
        if isinstance(cloud, PointCloud):
            points = cloud.cloud
        else:
            points = np.array(cloud)
            if points.ndim == 1:
                points = points.reshape((1, 3))

        transformed_redux = self.pca.transform(points[:, axes])
        untouched_axes = list(set(range(3)) - set(axes))
        transformed = np.insert(transformed_redux, untouched_axes, points[:, untouched_axes], axis=1)
        if isinstance(cloud, PointCloud):
            return PointCloud(transformed)
        elif original_shape is not None:
            return transformed.reshape(original_shape)
        return transformed

    def transform_inverse(self, cloud, axes=None):
        """
        Transforms from pca space to original space
        :param points: ndarray (N, 3) with the points to be transformed
        :return: ndarray(N, 3) with the transformed points
        """
        axes = axes or range(3)
        assert len(axes) == self.pca.n_components_, 'Expected len of axes to be \'{}\', found \'{}\' instead'.format(self.pca.n_components_, len(axes))
        points = cloud if not isinstance(cloud, PointCloud) else cloud.cloud

        transformed_redux = self.pca.inverse_transform(points[:, axes])
        untouched_axes = list(set(range(3)) - set(axes))
        transformed = np.insert(transformed_redux, untouched_axes, points[:, untouched_axes], axis=1)
        return PointCloud(transformed) if isinstance(cloud, PointCloud) else transformed


class Depth:
    def __init__(self, img, inverse_transform, pixel_radius, index, blur=2., mass_probability=None):
        self.img = img
        # Convert a pixel coordinate to a 2-dimensional world coordinate
        self.inverse_transform = inverse_transform
        self.index = index
        self.mass_probability = mass_probability

        self._apply_radius(pixel_radius)
        self.fill_missing()
        self.invert_distance()
        self.blur(blur)

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

    def invert_distance(self, uv=None):
        """Inverts the depth values of the pixels so that instead of representing the distance from the origin it
        shows the distance to the virtual camera or viceversa
        :param uv: tuple with row and column to be inverted. If none is supplied it inverts the whole object
        :return the inverted value if uv was supplied, None otherwise
        """
        if uv is None:
            self.img = self._invert_distance(self.img)
        else:
            return self._invert_distance(self.img[uv[0], uv[1]])

    def _invert_distance(self, value):
        _max = np.max(self.img)
        _min = np.min(self.img)
        return _max - value + _min

    def fill_missing(self):
        missing_idx = np.isinf(self.img)
        fill_value = np.min(self.img[np.logical_not(missing_idx)])
        self.img[missing_idx] = fill_value

    def to_object(self, uv):
        xy = self.inverse_transform(*uv)
        d = self._invert_distance(self.img[uv[0], uv[1]])
        return np.insert(xy, self.index, d)

    def entropy_score(self):
        entropy = 0
        for p in self.mass_probability:
            if p == 0:
                continue  # 0 * log(0) = 0
            entropy = p * np.log2(p)
        return -entropy


class PointCloud:
    def __init__(self, cloud):
        self.cloud = cloud
        self.pca_ = None

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

    def front_depth(self, approach_direction=1, shape=300):
        return self.to_depth(shape, 0)

    def right_depth(self, approach_direction=1, shape=300):
        return self.to_depth(shape, 1)

    def top_depth(self, approach_direction=1, shape=300):
        return self.to_depth(shape, 2)

    def orthographic_depth(self, shape=300):
        return [self.front_depth(shape=shape), self.right_depth(shape=shape), self.top_depth(shape=shape)]

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

    def orient_to_camera(self, camera_position):
        raise NotImplemented

    def to_depth(self, shape=300, index=0, padding=7, pixel_radius=3, blur=2.):
        """
        Construct depth image from point cloud. Index axis points towards the direction the camera is aiming (i.e.
        negative values occlude positive
        :param shape: output shape of the depth image (only one value, output is squared)
        :param index: column index in pc that defines depth
        :param padding: number of pixels to pad the image (same for all sides)
        :param pixel_radius: radius in pixels of each point in the cloud
        :param blur: blur coefficient for generating the final depth image
        :return: ndarray of shape shape with depth information
        """
        cloud = np.copy(self.cloud)
        assert pixel_radius <= padding

        inner_shape = shape - 2 * padding  # Shape of the image without the padding
        depth = np.ones((shape, shape)) * -np.inf  # Initial depth image
        spatial_idx = np.delete(range(3), index)  # Indices of the axes that represent the rows and columns of the image
        pixel_size = self.pixel_size(inner_shape)  # Number of world units per pixel (same for the 3 projections)

        _min = np.min(cloud[:, spatial_idx], axis=0)
        _max = np.max(cloud[:, spatial_idx], axis=0)
        cloud_center = _min + (_max - _min) / 2.  # Cloud center in object coordinates

        cloud[:, spatial_idx] -= cloud_center  # Center the cloud around 0

        image_center = np.floor(shape / 2.)
        image_center = np.array([image_center, image_center])

        mass_probability = np.zeros(shape*shape).astype(np.float)

        for p in cloud:
            c, r = (p[spatial_idx] / pixel_size + image_center).astype(np.int)
            if index == 0:  # -z is row and y is col
                r = shape - 1 - r
            elif index == 1:  # -z is row and -x is col
                r = shape - 1 - r
                c = shape - 1 - c
            else:  # x is col and -y is row
                r = shape - 1 - r
            depth[r, c] = max(depth[r, c], p[index])
            mass_probability[c + shape * r] += 1

        mass_probability /= np.sum(mass_probability).astype(np.float)

        if index == 0:
            inverse_transform = lambda r, c: np.array([
                (c - image_center[0]) * pixel_size + cloud_center[0],
                (shape - 1 - r - image_center[1]) * pixel_size + cloud_center[1]
            ]).astype(np.float)
        elif index == 1:
            inverse_transform = lambda r, c: np.array([
                (shape - 1 - c - image_center[0]) * pixel_size + cloud_center[0],
                (shape - 1 - r - image_center[1]) * pixel_size + cloud_center[1]
            ]).astype(np.float)
        else:
            inverse_transform = lambda r, c: np.array([
                (c - image_center[0]) * pixel_size + cloud_center[0],
                (shape - 1 - r - image_center[1]) * pixel_size + cloud_center[1]
            ]).astype(np.float)

        return Depth(depth, inverse_transform, pixel_radius, index, blur, mass_probability=mass_probability)

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
        assert self.cloud.size > 0, 'Can\'t do PCA on an empty cloud'
        axes = axes or range(3)
        transform = Transform(PCA().fit(self.cloud[:, axes]))
        return transform.transform(self, axes=axes), transform

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
    def __init__(self, model_fn=None):
        self.cloud = None
        self.network = None
        if model_fn is not None:
            from core.network import Network
            self.network = Network(model_fn=model_fn)

    def predict(self, cloud, predictor, roi=None):
        """
        Yields a point and orientation for a grasp in the point cloud
        :param cloud: Point cloud (ndarray of shape N, 3)
        :param predictor: callback to predict position and angle on a depth image
        :param roi: ROI in the cloud w.r.t work surface
        :return: position, orientation
        """
        roi = roi or [-np.inf, np.inf] * 3

        # Forward transform
        camera_cloud = PointCloud(cloud)  # w.r.t camera frame
        camera_position = np.array([0., 0., 0.]).reshape((1, 3))  # Position of the camera w.r.t. camera frame
        camera_frame = np.eye(3)  # Rows are the frame axes
        # render_pose(camera_cloud, camera_position, camera_orientation, camera_x, 1)
        # render_frame(camera_position, camera_frame[:,0], camera_frame[:, 1], camera_frame[:, 2])
        while True:
            try:
                plane_cloud = camera_cloud.find_plane()  # Largest plane w.r.t camera frame
                plane_cloud, tf_camera_to_plane = plane_cloud.pca()  # Largest plane w.r.t plane itself
                camera_position_plane = tf_camera_to_plane.transform(camera_position)  # Camera position w.r.t. the plane frame of reference
                camera_frame_plane = tf_camera_to_plane.transform(camera_frame)  # Camera orientation w.r.t. the plane frame of reference
                table_cloud = tf_camera_to_plane.transform(camera_cloud)
                # render_pose(table_cloud, camera_position_plane, camera_orientation_plane, camera_y_plane, 1, True)
                render_frame(camera_position_plane, camera_frame_plane[0], camera_frame_plane[1], camera_frame_plane[2], cloud=table_cloud)
                # render_frame(camera_position_plane, camera_frame_plane[0], camera_frame_plane[1],
                #              camera_frame_plane[2], wait=True)
                roi_cloud = table_cloud.filter_roi(roi)  # table_cloud points within the ROI
                object_cloud = roi_cloud.remove_plane()  # roi_cloud without the plane
                object_cloud, tf_roi_to_object = object_cloud.pca(axes=[0, 1])  # object_cloud with same z as table but x,y oriented by the object
                camera_position_object = tf_roi_to_object.transform(camera_position_plane, axes=[0, 1])  # Camera position w.r.t FoR of the object
                camera_frame_object = tf_roi_to_object.transform(camera_frame_plane, axes=[0, 1]) # Camera orientation w.r.t. FoR of the object
            except AssertionError as e:
                print('Caught error: {}, retrying'.format(e))
                continue
            break

        eye = np.eye(3)
        render_frame(camera_position_object, camera_frame_object[0], camera_frame_object[1], camera_frame_object[2],
                     cloud=object_cloud)
        render_frame([0, 0, 0], eye[0], eye[1], eye[2], wait=False, cloud=object_cloud)
        # Prediction
        # d1 = np.sign(np.dot(camera_orientation_object, [1, 0, 0]))
        # d2 = np.sign(np.dot(camera_orientation_object, [0, 1, 0]))

        (front_cloud, fidx, f_rotated), \
        (side_cloud, sidx, s_rotated), \
        (top_cloud, tidx, t_rotated) = orient_object_cloud(object_cloud, camera_position_object, camera_frame_object)

        was_rotated = [f_rotated, s_rotated, t_rotated]

        depths = [front_cloud.to_depth(index=fidx),
                  side_cloud.to_depth(index=sidx),
                  top_cloud.to_depth(index=tidx)]
        # TODO: what if PCA gives downwards vector after RANSAC?
        positions, zs, ys, widths, scores = [], [], [], [], []

        render_frame([0, 0, 0], eye[0], eye[1], eye[2], wait=False, cloud=front_cloud)

        for index, depth in enumerate(depths):
            position, z, y, width = predictor(depth, depth.index)
            print('Entropy: {}'.format(depth.entropy_score()))
            if was_rotated[index]:
                # Since z will never be rotated, we will always be rotating point and z around object z
                position[:2] *= -1
                z[:2] *= -1
                # Mirror the angle. Z is common for front and left, so we can always invert that one
                y[2] *= -1
            render_pose(object_cloud, position, z, y, width)

            # Backwards transform
            roi_position = tf_roi_to_object.transform_inverse(position.reshape((1, 3)), axes=[0, 1])
            roi_orientation = tf_roi_to_object.transform_inverse(z.reshape((1, 3)), axes=[0, 1])
            roi_y = tf_roi_to_object.transform_inverse(y.reshape((1, 3)), axes=[0, 1])

            camera_position = tf_camera_to_plane.transform_inverse(roi_position)
            camera_orientation = tf_camera_to_plane.transform_inverse(roi_orientation)
            camera_x = tf_camera_to_plane.transform_inverse(roi_y)
            roi_com = tf_camera_to_plane.transform_inverse(np.zeros((1, 3)))
            camera_orientation -= roi_com
            camera_x -= roi_com
            camera_orientation = camera_orientation / np.linalg.norm(camera_orientation)
            camera_x = camera_x / np.linalg.norm(camera_x)

            # render_pose(camera_cloud, camera_position, camera_orientation, camera_y, width)

            positions.append(camera_position)
            zs.append(camera_orientation/np.linalg.norm(camera_orientation))
            ys.append(camera_x/np.linalg.norm(camera_x))
            widths.append(width)
            scores.append(depth.entropy_score())

        return positions, zs, ys, widths, scores

    def network_predictor(self, depth_img, index, debug=True):
        from core.network import get_grasps_from_output
        positions, angles, widths = self.network.predict(depth_img.img)
        gs = get_grasps_from_output(positions, angles, widths)
        if debug:
            from core.network import get_output_plot
            get_output_plot(depth_img.img, positions, angles, widths)
            plt.ion()
            plt.show()
            plt.pause(.1)
        assert len(gs) > 0
        grasp = gs[0]
        point = depth_img.to_object(grasp.center)

        delta = np.array([np.cos(grasp.angle), np.sin(grasp.angle)]) * grasp.width / 2.
        start = np.subtract(grasp.center, delta).astype(np.int)
        end = np.add(grasp.center, delta).astype(np.int)
        img_axes = np.delete(range(3), index)
        width = np.abs(np.linalg.norm(depth_img.to_object(end)[img_axes] - depth_img.to_object(start)[img_axes]))

        y = np.subtract(end, start)
        y = y / np.linalg.norm(y)
        y = np.insert(y, index, 0)

        z = np.insert(np.zeros(2), index, 1)
        return point, z, y, width

    @staticmethod
    def manual_predictor(depth_img, index):
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
        cv2.destroyAllWindows()

        global start, end
        img_axes = np.delete(range(3), index)
        width = np.abs(np.linalg.norm(depth_img.to_object(end[::-1])[img_axes] - depth_img.to_object(start[::-1])[img_axes]))
        z = np.insert(np.zeros(2), index, -1)
        # width = 0

        start = np.array(start)
        start[1] = 299 - start[1]
        end = np.array(end)
        end[1] = 299 - end[1]
        # If picture is taken from y axis horizontal axis goes from right to left
        if index == 1:
            start[0] = 299 - start[0]
            end[0] = 299 - end[0]
        # angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        # print('Angle {}'.format(angle))

        y = np.insert([end[0] - start[0], end[1] - start[1]], index, 0)
        y = y / np.linalg.norm(y)

        return point, z, y, width


def render_pose(cloud, position, z, y, width, wait=False):
    cloud = cloud.cloud
    z *= float(width)
    y *= width/2.
    # y *= float(width)
    z_axis = np.linspace(position, position + z, 1e3).squeeze()
    # y_axis = np.linspace(position, position + y, 1e3).squeeze()
    y_axis = np.linspace(position - y, position + y, 1e3).squeeze()
    blue = np.repeat([[0, 0, 1]], z_axis.shape[0], axis=0)
    green = np.repeat([[0, 1, 0]], y_axis.shape[0], axis=0)
    colors = np.concatenate((np.ones(cloud.shape), blue, green))
    points = np.concatenate((cloud, z_axis, y_axis))
    viewer = pptk.viewer(points)
    viewer.attributes(colors)
    viewer.set(point_size=0.0005)

    if wait:
        raw_input('Enter to continue')


def orient_object_cloud(cloud, camera_position, camera_frame):
    camera_z = (camera_frame[2] - camera_position).flatten()

    # Find index of front and side axes (based on cosine between object cloud and camera vector)
    side_idx, front_idx = np.argsort(np.abs(camera_z[:2]))
    top_idx = 2

    # Front view
    front_cloud = PointCloud(np.copy(cloud.cloud))
    front_rotated = camera_z[front_idx] > 0
    if front_rotated:
        print('Front view was rotated')
        front_cloud.cloud[:, :2] *= -1  # Rotate 180 around z
    # Side view
    side_cloud = PointCloud(np.copy(cloud.cloud))
    side_rotated = camera_z[side_idx] > 0
    if side_rotated:
        print('Side view was rotated')
        side_cloud.cloud[:, :2] *= -1  # Rotate 180 around z
    # Top view (no need to rotate)
    top_cloud = cloud

    return (front_cloud, front_idx, front_rotated), (side_cloud, side_idx, side_rotated), (top_cloud, top_idx, False)


def render_frame(position, x, y, z, wait=False, cloud=None):
    x_axis = np.linspace(position, x, 1e3).squeeze()
    y_axis = np.linspace(position, y, 1e3).squeeze()
    z_axis = np.linspace(position, z, 1e3).squeeze()

    r = np.repeat([[1, 0, 0]], x_axis.shape[0], axis=0)
    g = np.repeat([[0, 1, 0]], y_axis.shape[0], axis=0)
    b = np.repeat([[0, 0, 1]], z_axis.shape[0], axis=0)

    colors = np.concatenate((r, g, b))

    all_points = np.concatenate((x_axis, y_axis, z_axis))

    if cloud is not None:
        cloud_points = cloud.cloud
        all_points = np.concatenate((cloud_points, all_points))
        w = np.repeat([[1, 1, 1]], cloud_points.shape[0], axis=0)
        colors = np.concatenate((w, colors))

    viewer = pptk.viewer(all_points)
    viewer.attributes(colors)
    viewer.set(point_size=0.0005)

    if wait:
        viewer.wait()


def render(cloud, wait=False):
    pptk.viewer(cloud.cloud)
    if wait:
        raw_input('key to continue')


if __name__ == '__main__':
    import pylab as plt
    cloud = PointCloud.from_npy('../test/points.npy')
    onet = OrthoNet(model_fn='/Users/mario/Developer/msc-thesis/data/networks/beam_search_transpose/arch_C9x9x32_C5x5x32_C5x5x16_C3x3x8_C3x3x8_T3x3x8_T3x3x8_T5x5x16_T9x9x32_depth_3_model.hdf5')
    points, orientations, angles, widths, scores = onet.predict(cloud.cloud, onet.network_predictor,roi=[-2, 1, -.15, .25, 0, 0.2])
    # points, orientations, angles, widths, scores = onet.predict(cloud.cloud, onet.manual_predictor, roi=[-2, 1, -.15, .25, 0, 0.2])

    plt.pause(1e5)
    raw_input('Press ENTER to quit')

    # marker = np.linspace(point, point + orientation, 1e3).squeeze()
    # render_prediction(cloud, marker)

    # # points = np.linspace([-width/2., 0, 0], [width/2., 0, 0], int(10e3))
    # # r = R.from_rotvec(orientation * angle)
    # # points = r.apply(points)
    # # points += point
    # points = grasp_marker(point, orientation, angle, width)
    # # render_prediction(cloud, points)
