from unittest import TestCase
from simulator.simulator import Simulator
import numpy as np


class TestSimulatorCamera(TestCase):
    def test_move_gripper(self):
        sim = Simulator(gui=True, debug=False, use_egl=False)

        sim.add_gripper('../simulator/gripper.urdf')
        sim.load('../simulator/data/cube_small.urdf')
        pos = [0, 0, 0.05]
        pose = np.append(pos, np.radians([0, 0, 0]))
        sim.run(epochs=240)
        sim.set_gripper_width(0.07)
        sim.move_gripper_to(pose, angvel=4, linvel=0.6)
        sim.close_gripper()
        pose[2] = 0.5
        sim.move_gripper_to(pose, angvel=4, linvel=0.6)

        sim.run(epochs=500)

        # TODO: move_to is completely useless for 6DOF poses

    def test_network(self):
        import matplotlib
        matplotlib.use('Agg')
        from core import network as net
        network = net.Network(model_fn='../data/networks/ggcnn_rss/epoch_29_model.hdf5')
        sim = Simulator(gui=True, debug=False, use_egl=False)
        sim.load('../simulator/data/cube_small.urdf')
        sim.run(epochs=1000)

        _, depth = sim.cam.snap()
        position, angle, width = network.predict(depth)
        gs = net.get_grasps_from_output(position, angle, width, 1)[0]
        pose, gripper_width = sim.cam.compute_grasp(gs.as_bb.points, depth[gs.center])
        pose = np.concatenate((pose, [0, 0, gs.angle]))
        result = sim.evaluate_grasp(pose, gripper_width)
        print '\nResult: {}'.format(result)

        del sim
        net.save_output_plot(depth, position, angle, width, 'test_network.png', 1)
