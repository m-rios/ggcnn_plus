#!/usr/bin/env python
import argparse
from simulator import Simulator
import os
import glob
import pybullet as p
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('logfile', help='path to either a log file or a folder with log files')
args = parser.parse_args()

SCENES_PATH = os.environ['GGCNN_SCENES_PATH']


if os.path.isfile(args.logfile):
    log_fns = [args.logfile]
else:
    log_fns = glob.glob(os.path.join(args.logfile, '*.log'))

sim = Simulator(gui=False)
for log_fn in log_fns:
    scene_name = log_fn.split('/')[-1].split('.')[-2]
    scene_fn = os.path.join(SCENES_PATH, scene_name + '_scene.csv')
    sim.replay(log_fn, scene_fn)

