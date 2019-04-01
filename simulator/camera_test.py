import glob
import os
from simulator import Simulator
from time import sleep
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #GUI = True
    #sim = Simulator(gui=GUI, g=-10, debug=False, epochs=240)
    #sim.load('cube_small.urdf', pos=[0.2, 0.5,0.025])
    #sim.cam.target = [0.2, 0.5, 0.0]
    #sim.cam.project(sim.cam.target)
    #rgb,depth = sim.cam.snap()
    #if not GUI:
    #    plt.imshow(depth)
    #    plt.show()
    #u = 146
    #v = 151
    #d = depth[v, u]
    ##u = 201
    ##v = 155
    ##d = depth[v, u]
    #pt = sim.cam.world_from_camera(u, v, d)
    #sim.drawFrame(pt)
    #sim.run(epochs=100000, autostop=False)
    #exit(0)

    GUI = True
    sim = Simulator(gui=GUI, g=-10, debug=False, epochs=240)
    sim.restore("scenes/871dcd6740ed8d2891a7ac029a825f73_scene.csv", "models")
    rgb,depth = sim.cam.snap()
    if not GUI:
        plt.imshow(depth)
        plt.show()
    u = 182
    v = 226
    d = depth[v, u]
    #u = 201
    #v = 155
    #d = depth[v, u]
    pt = sim.cam.world_from_camera(u, v, d)
    sim.drawFrame(pt)
    sim.run(epochs=100000, autostop=False)

