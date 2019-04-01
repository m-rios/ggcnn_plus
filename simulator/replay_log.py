import argparse
from simulator import Simulator
import os
import glob
import pybullet as p

parser = argparse.ArgumentParser()
parser.add_argument('logfile', help='path to either a log file or a folder with log files')
parser.add_argument('--video', action='store_true', help='convert logfile(s) to mp4')
args = parser.parse_args()

SCENES_PATH = os.environ['GGCNN_SCENES_PATH']


if os.path.isfile(args.logfile):
    log_fns = [args.logfile]
else:
    log_fns = glob.glob(os.path.join(args.logfile, '*.log'))

sim = Simulator(gui=True)
p.resetDebugVisualizerCamera(2., 0., -60, [0, 0, 0])
for log_fn in log_fns:
    if args.video:
        output_log = log_fn.replace('.log', '.mp4')
        log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, output_log)
    scene_name = log_fn.split('/')[-1].split('.')[-2]
    scene_fn = os.path.join(SCENES_PATH, scene_name + '_scene.csv')
    sim.replay(log_fn, scene_fn)
    if args.video:
        p.stopStateLogging(log)

