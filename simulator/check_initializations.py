from simulator import Simulator
import os
import glob

sim = Simulator(debug=True,timestep=1./240., gui=True, g=-10)

scene_fns = glob.glob('/Users/mario/Desktop/scenes/*.csv')

for scene_fn in scene_fns:
    sim.restore(scene_fn, '/Volumes/Peregrine/obj_test')
    sim.run(epochs=100)

