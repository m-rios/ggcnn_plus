#!/usr/bin/env python
import argparse
from simulator import Simulator
import os
import glob
import pybullet as p
import cv2
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('logfile', help='path to either a log file or a folder with log files')
parser.add_argument('--scenes', default=os.environ['GGCNN_SCENES_PATH'], help='path to either a log file or a folder with log files')
args = parser.parse_args()


if os.path.isfile(args.logfile):
    log_fns = [args.logfile]
else:
    log_fns = glob.glob(os.path.join(args.logfile, '*.log'))

scenes = h5py.File(args.scenes, 'r')
sim = Simulator(gui=True)
for log_fn in log_fns:
    scene_name = log_fn.split('/')[-1].split('.')[-2]
    #scene_fn = os.path.join(args.scenes, scene_name + '_scene.csv')
    scene_idx = np.where( scenes['name'][:] == scene_name)[0][0]
    scene = scenes['scene'][scene_idx]
    sim.replay(log_fn, scene)

