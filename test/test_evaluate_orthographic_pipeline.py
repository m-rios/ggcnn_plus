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

        cloud = PointCloud(sim.cam.point_cloud())
        print 'cloud was generated'

        cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

        eye = np.eye(3)
        orthographic.render_frame([0, 0, 0], eye[0], eye[1], eye[2], cloud=cloud)

    def test_transform_camera_to_world(self):
        sim = Simulator(use_egl=False)
        scenes_ds = h5py.File('../data/scenes/shapenetsem40_5.hdf5', 'r')
        scene = scenes_ds['scene'][np.random.randint(len(scenes_ds['scene'][:]))]
        sim.restore(scene, os.environ['MODELS_PATH'])

        sim.cam.pos = [1, 1, 1.5]
        sim.cam.width = 200
        sim.cam.height = 200

        cloud_orig = PointCloud(sim.cam.point_cloud())
        print 'cloud was generated'

        cloud_transformed = ortho_pipeline.transform_world_to_camera(cloud_orig, sim.cam)
        cloud_recovered = ortho_pipeline.transform_camera_to_world(cloud_transformed, sim.cam)

        diff = np.abs(cloud_orig.cloud - cloud_recovered.cloud)

        assert np.all(diff < 1e-2)
