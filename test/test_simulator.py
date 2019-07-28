from unittest import TestCase

from simulator.simulator import Simulator

import h5py
import numpy as np
import os


class TestSimulatorCamera(TestCase):
    def test_move_gripper(self):
        sim = Simulator(gui=True, use_egl=False)

        sim.add_gripper('../simulator/gripper.urdf')
        pos = [0, 0, 1]
        pose = np.append(pos, np.radians([0, -90, 90]))
        sim.move_gripper_to(pose, angvel=4)
        sim.run(epochs=1000000)

        # TODO: move_to is completely useless for 6DOF poses

