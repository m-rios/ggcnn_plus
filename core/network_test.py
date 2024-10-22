from simulator.simulator import Simulator, VIDEO_LOGGER
import numpy as np
from core import network as net
import argparse
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
parser.add_argument('--scenes', default = os.environ['GGCNN_SCENES_PATH'] + '/shapenet_1.hdf5')
parser.add_argument('--scene_name', default=None)
parser.add_argument('--models', default = os.environ['MODELS_PATH'])
parser.add_argument('--gui', action='store_true')
parser.add_argument('--logvideo', action='store_true')

args = parser.parse_args()

scene_ds = h5py.File(args.scenes, 'r')
if args.scene_name is None:
    scene_idx = np.random.randint(0, 39)
else:
    scene_idx = int(np.where(scene_ds['name'][:] == args.scene_name)[0])
scene = scene_ds['scene'][scene_idx]

network = net.Network(model_fn=args.model)
sim = Simulator(gui=args.gui, timeout=4, debug=False, use_egl=False, timestep=1e-3)
sim.restore(scene, args.models)
depth = scene_ds['depth'][scene_idx]

position, angle, width = network.predict(depth)

if not args.gui:
    net.plot_output(depth, position, angle, width, 1)

gs = net.get_grasps_from_output(position, angle, width, 1)[0]
# Send grasp to simulator and evaluate
pose, width = sim.cam.compute_grasp(gs.as_bb.points, depth[gs.center])
print pose, width, gs.angle
pose = np.concatenate((pose, [0, 0, gs.angle]))
if args.logvideo:
    sim.start_log('/home/mario/Developer/msc-thesis/videolog.mp4', VIDEO_LOGGER)
result = sim.evaluate_grasp(pose, width)
if args.logvideo:
    sim.stop_log()


