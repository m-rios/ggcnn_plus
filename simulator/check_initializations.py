from simulator import Simulator
import argparse
import os
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--scene', default=os.environ['GGCNN_SCENES_PATH'] + '/test.hdf5')
parser.add_argument('--objpath', default=os.environ['MODELS_PATH'])
args = parser.parse_args()

sim = Simulator(debug=True,timestep=1e-4, gui=True, g=-10)

scenes = h5py.File(args.scene, 'r')

for scene in scenes['scene']:
    sim.restore(scene, args.objpath)
    sim.cam.snap()
    sim.run(epochs=1000)

