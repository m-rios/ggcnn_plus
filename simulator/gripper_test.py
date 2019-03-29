import glob
import os
import pybullet as p
from simulator import Simulator
from time import sleep

if __name__ == '__main__':

    sim = Simulator(timeout=20, gui=True, timestep=1e-4, g=-10, debug=True, epochs=240)

    sim.restore("scenes/1d4480abe9aa45ce51a99c0e19a8a54_scene.csv", "models2")
    sim.add_gripper('gripper.urdf')
    # Pregrasp
    print('Opening gripper')
    sim.open_gripper()
    print('Moving to pregrasp')
    sim.move_gripper_to([0.2, 0.1, 0.2, 0, 0, 1.5])
    sim.run(epochs=100, autostop=False)
    print('Moving to grasp')
    sim.move_gripper_to([0.2, 0.1, 0.05, 0, 0, 0.7])
    print('Closing gripper')
    sim.close_gripper()
    print('Moving to postgrasp')
    sim.move_gripper_to([0.2, 0.1, 0.5, 0, 0, 0.7])
