from simulator import Simulator
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('scenes_path')
parser.add_argument('--objpath', default=os.environ['SHAPENET_PATH'])
args = parser.parse_args()

sim = Simulator(debug=True,timestep=1e-4, gui=True, g=-10)

scene_fns = glob.glob(args.scenes_path + '/*.csv')

for scene_fn in scene_fns:
    sim.restore(scene_fn, args.objpath)
    sim.cam.snap()
    sim.run(epochs=1000)

