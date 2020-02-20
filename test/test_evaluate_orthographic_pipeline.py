from unittest import TestCase
from simulator.simulator import Simulator
from core.orthographic import PointCloud

import pylab as plt
import numpy as np
import os
import h5py
import pptk
import core.orthographic as orthographic
import core.evaluate_orthographic_pipeline as ortho_pipeline


class TestEvaluateOrthographicPipeline(TestCase):

    def test_transform_world_to_camera(self):
        sim = Simulator(use_egl=False)
        scenes_ds = h5py.File('../data/scenes/shapenetsem40_5.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        sim.cam.pos = [1, 1, 1.5]
        sim.cam.width = 600
        sim.cam.height = 600

        cloud = sim.cam.point_cloud()
        print 'cloud was generated'

        cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

        eye = np.eye(3)
        orthographic.render_frame([0, 0, 0], eye[0], eye[1], eye[2], cloud=PointCloud(cloud))

    def test_transform_camera_to_world(self):
        sim = Simulator(use_egl=False)
        scenes_ds = h5py.File('../data/scenes/shapenetsem40_5.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        sim.cam.pos = [1, 1, 1.5]
        sim.cam.width = 200
        sim.cam.height = 200

        cloud_orig = sim.cam.point_cloud()
        print 'cloud was generated'

        cloud_transformed = ortho_pipeline.transform_world_to_camera(cloud_orig, sim.cam)
        cloud_recovered = ortho_pipeline.transform_camera_to_world(cloud_transformed, sim.cam)

        diff = np.abs(cloud_orig - cloud_recovered)

        assert np.all(diff < 1e-2)

    def test_object_to_world(self):
        sim = Simulator(gui=False, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.cam.pos = [0, 0, 1.5]
        sim.cam.width = 300
        sim.cam.height = 300

        cloud = sim.cam.point_cloud()

        cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

        net = orthographic.OrthoNet(
            model_fn='/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')

        points, orientations, angles, widths, scores = net.predict(cloud, net.network_predictor,
                                                                   debug=False)
        best_idx = np.argmax(scores)

        p = points[best_idx]
        z = orientations[best_idx]
        x = angles[best_idx]
        w = widths[best_idx]
        y = np.cross(z, x)

        # Camera to world transformation
        p = ortho_pipeline.transform_camera_to_world(p, sim.cam)
        R = ortho_pipeline.get_camera_frame(sim.cam)
        z = R.dot(z.T).T
        x = R.dot(x.T).T

        recloud = ortho_pipeline.transform_camera_to_world(cloud, sim.cam)
        orthographic.render_pose(PointCloud(recloud), p, z, x, w)

    def test_top_grasp(self):
        sim = Simulator(gui=True, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.cam.pos = [0, 0, 1.5]
        sim.cam.width = 600
        sim.cam.height = 600

        cloud = sim.cam.point_cloud()

        cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

        net = orthographic.OrthoNet(model_fn='/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')

        ps, zs, xs, ws, scores = net.predict(cloud, net.network_predictor, debug=True, predict_best_only=True)

        best_idx = np.argmax(scores)
        print best_idx, ps, scores

        p = ortho_pipeline.transform_camera_to_world(ps[best_idx], sim.cam)
        R = ortho_pipeline.get_camera_frame(sim.cam)
        z = R.dot(zs[best_idx].T).T
        x = R.dot(xs[best_idx].T).T
        w = ws[best_idx]

        sim.add_gripper('../simulator/gripper.urdf')
        sim.add_debug_pose(p, z, x, w)
        sim.teleport_to_pre_grasp(p, z, x, w)
        sim.grasp_along(z)
        sim.move_to_post_grasp()
        sim.run(1000)

    def test_obstacle_avoidance(self):
        sim = Simulator(gui=False, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.cam.pos = [0, 0, 1.5]
        sim.cam.width = 100
        sim.cam.height = 100

        cloud = sim.cam.point_cloud()

        cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

        net = orthographic.OrthoNet(
            model_fn='/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5')

        ps, zs, xs, ws, scores = net.predict(cloud, net.network_predictor, debug=True, predict_best_only=True, n_attempts=5)

        best_idx = np.argmax(scores)
        print best_idx, ps, scores

        p = ortho_pipeline.transform_camera_to_world(ps[best_idx], sim.cam)
        R = ortho_pipeline.get_camera_frame(sim.cam)
        z = R.dot(zs[best_idx].T).T
        x = R.dot(xs[best_idx].T).T
        w = ws[best_idx]

        sim.add_gripper('../simulator/gripper.urdf')
        sim.add_debug_pose(p, z, x, w)
        sim.teleport_to_pre_grasp(p, z, x, w)
        sim.grasp_along(z)
        sim.move_to_post_grasp()
        sim.run(1000)
