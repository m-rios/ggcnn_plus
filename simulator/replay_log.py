import argparse
from simulator import Simulator
import os

parser = argparse.ArgumentParser()
parser.add_argument('logfile', help='path to log file')
args = parser.parse_args()

SCENES_PATH = os.environ['GGCNN_SCENES_PATH']
scene_name = args.logfile.split('/')[-1].split('.')[-2]

scene_fn = os.path.join(SCENES_PATH, scene_name + '_scene.csv')

sim = Simulator(gui=True)
sim.replay(args.logfile, scene_fn)

