from core.orthographic import OrthoNet, PointCloud
from simulator.simulator import Simulator

import os
import h5py
import numpy as np
import core.orthographic as orthographic


def transform_camera_to_world(cloud, camera):
    R = get_camera_frame(camera)
    t = np.array(camera.pos)[:, np.newaxis]
    return PointCloud((R.dot(cloud.cloud.T) + t).T)


def transform_world_to_camera(cloud, camera):
    R = get_camera_frame(camera).T
    t = R.dot(-np.array(camera.pos))[:, np.newaxis]
    return PointCloud((R.dot(cloud.cloud.T) + t).T)


def get_camera_frame(camera):
    """
    Computes the camera rotation matrix w.r.t world coordinates from a given camera instance
    :param camera: Sim camera instance
    :return: Transformation matrix representing the position and orientation of the camera
    """

    cW = np.array(camera.pos)  # Camera position wrt world
    tW = np.array(camera.target)  # Camera target wrt world

    # Basis of camera frame wrt world
    zW = tW - cW
    zW /= np.linalg.norm(zW)
    if np.linalg.norm(zW.flatten()[:2]) < 0.001:
        # If z is almost vertical, x is aligned with world's x
        xW = np.array([1, 0, 0])
    else:
        # Otherwise x is in the XY plane and orthogonal to z
        xW = np.array([zW[1], -zW[0], 0])
        xW = xW / np.linalg.norm(xW)
    # Right handed frame, y is computed from the other known axes
    yW = np.cross(zW, xW)

    return np.column_stack((xW, yW, zW))


if __name__ == '__main__':
    sim = Simulator(use_egl=False)
    scenes_ds = h5py.File('../data/scenes/shapenetsem40_5.hdf5', 'r')
    scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
    sim.restore(scene, os.environ['MODELS_PATH'])

    sim.cam.pos = [1, 1, 1.5]
    sim.cam.width = 600
    sim.cam.height = 600

    cloud = PointCloud(sim.cam.point_cloud())
    print 'cloud was generated'

    cloud = transform_world_to_camera(cloud, sim.cam)

    eye = np.eye(3)
    orthographic.render_frame([0, 0, 0], eye[0], eye[1], eye[2], cloud=cloud)

    onet = OrthoNet(model_fn='/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')
    points, orientations, angles, widths, scores = onet.predict(cloud.cloud, onet.manual_predictor)
