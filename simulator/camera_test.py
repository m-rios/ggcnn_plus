import glob
import os
from simulator import Simulator
from time import sleep
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    sim = Simulator(gui=False, g=-10, debug=True, epochs=240)

    sim.restore("scenes/1d4480abe9aa45ce51a99c0e19a8a54_scene.csv", "models2")
    sim.load('cube_small.urdf', pos=[0.9,0.9,0.1])
    rgb,depth = sim.cam.snap()
    plt.imshow(depth)
    plt.pause(0)
    u = 170
    v = 111
    d = depth[v, u]
    #u = 201
    #v = 155
    #d = depth[v, u]
    pt = sim.cam.world_from_camera(u, v, d)
    sim.drawFrame(pt)
    sim.run(epochs=100000, autostop=False)

