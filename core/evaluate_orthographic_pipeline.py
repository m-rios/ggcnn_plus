from core.orthographic import OrthoNet, PointCloud
from simulator.simulator import Simulator

import os
import h5py
import numpy as np
import core.orthographic as orthographic


def transform_world_to_camera(cloud, camera):
    """
    Rotates a cloud wrt world so that it is wrt camera
    """
    cW = np.array(camera.pos)  # Camera position wrt world
    tW = np.array(camera.target)  # Camera target wrt world

    # Basis of camera frame wrt world
    zW = tW - cW
    zW /= np.linalg.norm(zW)
    xW = np.cross(zW, np.array([0, 0, 1]))
    yW = np.cross(zW, xW)

    # # End points of the basis (for debugging purposes)
    # xeW = cW + xW
    # yeW = cW + yW
    # zeW = cW + zW
    # orthographic.render_frame(cW, xeW, yeW, zeW, cloud=cloud)

    Rcw = np.column_stack((xW, yW, zW))  # Rotation from camera to world
    Rwc = Rcw.T  # Rotation from world to camera

    rotated_cloud = Rwc.dot(cloud.cloud.T)
    transformed_cloud = PointCloud((rotated_cloud - Rwc.dot(cW[:, np.newaxis])).T)

    # eye = np.eye(3)
    # orthographic.render_frame([0, 0, 0], eye[0], eye[1], eye[2], cloud=transformed_cloud)

    return transformed_cloud


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
