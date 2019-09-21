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
parser.add_argument('scenes', help='path to hdf5 scenes file')
parser.add_argument('--models', default=os.environ['MODELS_PATH'])
parser.add_argument('--convert', action='store_true', help='convert to mp4 instead of replaying')
args = parser.parse_args()


if os.path.isfile(args.logfile):
    log_fns = [args.logfile]
else:
    log_fns = glob.glob(os.path.join(args.logfile, '*.log'))

scenes = h5py.File(args.scenes, 'r')
sim = Simulator(gui = not args.convert)
for log_fn in log_fns:
    scene_name = log_fn.split('/')[-1].split('.')[-2]
    scene_idx = np.where( scenes['name'][:] == scene_name)[0][0]
    scene = scenes['scene'][scene_idx]
    if args.convert:
        video_fn = log_fn.replace('.log', '.mp4')
        sim.start_log(video_fn)
    sim.replay(log_fn, scene, args.models)
    if args.convert:
        sim.stop_log()

