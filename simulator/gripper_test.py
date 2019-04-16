import glob
import os
import pybullet as p
from simulator import Simulator
from time import sleep
import numpy as np
import pylab as plt

if __name__ == '__main__':

    sim = Simulator(timeout=20, gui=True, timestep=1e-4, g=-10, debug=True, epochs=240)
    SCENES_PATH = os.environ['GGCNN_SCENES_PATH']
    MODELS_PATH = os.environ['MODELS_PATH']

    sim.add_gripper('gripper.urdf')
    fs, zs, ts = sim.move_test([0.1, 0.1, 0.3, 0, 0, 0])
    zs0 = np.array(zs) * 10000
    fs0 = np.array([np.array(x) for x in fs])
    ts0 = np.array(ts)
    sim.run(epochs=1000)
    fs, zs, ts = sim.move_test([0, 0, -1, 0, 0, 0])
    zs = np.array(zs) * 10000
    fs = np.array([np.array(x) for x in fs])
    ts = np.array(ts)
    zs = np.concatenate((zs0, zs))
    fs = np.concatenate((fs0, fs))
    ts = np.concatenate((ts0, ts))
    plt.plot(zs, 'k')
    plt.plot(fs[:,0], 'r')
    plt.plot(fs[:,1], 'g')
    plt.plot(fs[:,2], 'b')
    plt.plot(ts, 'm')
    plt.show()

