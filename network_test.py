from keras.models import load_model
from skimage.filters import gaussian
from ggcnn.dataset_processing.grasp import detect_grasps, BoundingBoxes
from simulator.simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
import network as net
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ggcnn/data/networks/ggcnn_rss/epoch_29_model.hdf5')
parser.add_argument('--scene', default = os.environ['GGCNN_SCENES_PATH'] + '/1d4480abe9aa45ce51a99c0e19a8a54_scene.csv')
parser.add_argument('--gui', action='store_true')

args = parser.parse_args()

network = net.Network(args.model)
sim = Simulator(gui=args.gui, timeout=4, debug=True)
sim.restore(args.scene, os.environ['MODELS_PATH'])
rgb, depth = net.read_input_from_scenes([args.scene], 300, 300)

position, angle, width = network.predict(depth)

if not args.gui:
    net.plot_output(depth, position, angle, width, 1)

gs = net.get_grasps_from_output(position, angle, width, 1)[0]
# Send grasp to simulator and evaluate
print 'Pixels: ' + str(gs.center)
pose, width = sim.cam.compute_grasp(gs.as_bb.points, depth[0][gs.center])
print 'Width: ' + str(width)
print 'Pose: ' + str(pose)
pose = np.concatenate((pose, [0, 0, gs.angle]))
result = sim.evaluate_grasp(pose, width)

