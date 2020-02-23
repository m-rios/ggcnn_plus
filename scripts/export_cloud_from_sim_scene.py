from simulator.simulator import Simulator
import h5py
import pptk
import numpy as np
import core.evaluate_orthographic_pipeline as ortho_pipeline

SCENE_FN = '/Users/mario/Developer/msc-thesis/data/scenes/200220_1700_manually_generated_scenes.hdf5'
scene = h5py.File(SCENE_FN)['scene'][0]
sim = Simulator(use_egl=False, gui=False)
sim.cam.pos = [.6, 0., .3]
sim.restore(scene, '../data/3d_models/shapenetsem40')
cloud = sim.cam.point_cloud()

cloud = ortho_pipeline.transform_world_to_camera(cloud, sim.cam)

pptk.viewer(cloud)
np.save('../test/horizontal_bottle.npy', cloud)
