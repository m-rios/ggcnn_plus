from unittest import TestCase
from simulator.simulator import Simulator
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R


class TestSimulatorCamera(TestCase):
    def test_move_gripper(self):
        sim = Simulator(gui=True, debug=False, use_egl=False)

        sim.add_gripper('../simulator/gripper.urdf')
        sim.load('../simulator/data/cube_small.urdf')
        pos = [0, 0, 0.05]
        # pos = [0, 0, 0.02]
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

    def test_forcetorquesensor(self):
        import pybullet as p
        p.connect(p.DIRECT)
        hinge = p.loadURDF("../simulator/data/hinge.urdf")
        print("mass of linkA = 1kg, linkB = 1kg, total mass = 2kg")

        hingeJointIndex = 0

        # by default, joint motors are enabled, maintaining zero velocity
        p.setJointMotorControl2(hinge, hingeJointIndex, p.VELOCITY_CONTROL, 0, 0, 0)

        p.setGravity(0, 0, -10)
        p.stepSimulation()
        print("joint state without sensor")

        print(p.getJointState(0, 0))
        p.enableJointForceTorqueSensor(hinge, hingeJointIndex)
        p.stepSimulation()
        print("joint state with force/torque sensor, gravity [0,0,-10]")
        print(p.getJointState(0, 0))
        p.setGravity(0, 0, 0)
        p.stepSimulation()
        print("joint state with force/torque sensor, no gravity")
        print(p.getJointState(0, 0))

        p.disconnect()

    def test_close_gripper(self):
        sim = Simulator(gui=True, use_egl=False)
        sim.add_gripper('../simulator/gripper.urdf')
        sim.run(epochs=100)
        result = sim.close_gripper()
        # result = np.array(result)
        # import pylab as plt
        #
        # plt.plot(result[:, 1], '-r')
        # # plt.plot(result[:, 0], '-g')
        # plt.show()

    def test_gripper_autostop(self):
        sim = Simulator(gui=True, use_egl=False)
        sim.add_gripper('../simulator/gripper.urdf')
        result = sim.move_gripper_to([0, 0, 0, 0, 0, 0])

        # import pylab as plt
        # plt.plot(result, '-r')
        # plt.show()

    def test_static_gripper(self):
        scene = h5py.File('../data/scenes/shapenetsem40_5.hdf5')['scene'][31]
        sim = Simulator(gui=True, use_egl=False)
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.add_gripper('../simulator/gripper.urdf')
        sim.load('../simulator/bin.urdf', pos=sim.bin_pos)
        sim.run(epochs=50000)

    def test_teleport_to_pose(self):
        sim = Simulator(gui=True, use_egl=False, debug=True)
        sim.add_gripper('../simulator/gripper.urdf')

        sim.run_debug_teleport()

    def test_teleport_rotation_from_axis(self):
        sim = Simulator(gui=True, use_egl=False)
        sim.add_gripper('../simulator/gripper.urdf')
        width = .2
        angles = np.random.rand(3) * 2 * np.pi - np.pi
        target_rot = R.from_euler('ZXZ', angles)
        debug_rot = target_rot.as_dcm()
        sim.add_debug_pose([0, 0, 1], debug_rot[:, 2], debug_rot[:, 0], width)
        sim.run(epochs=100)
        gripper_rot = target_rot * R.from_euler('X', np.pi)
        sim.teleport_to_pose([0, 0, 1], gripper_rot.as_euler('XYZ'), width)
        sim.run(epochs=10000)

    def test_add_debug_pose(self):
        sim = Simulator(gui=True, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')

        cam_rot = R.from_euler('ZXZ', [0, 90, -30], degrees=True).as_dcm()
        print cam_rot
        sim.add_debug_pose([0, 0, 1], cam_rot[:,2], cam_rot[:,0], .5)
        sim.run()

    def test_teleport_to_pre_grasp(self):
        sim = Simulator(gui=True, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.add_gripper('../simulator/gripper.urdf')
        p = [-.032, -.032, .1]
        z = [1, 1, 0]
        x = [1, -1, 0]
        w = .1
        sim.add_debug_pose(p, z, x, w)
        sim.run(epochs=100)
        sim.teleport_to_pre_grasp(p, z, x, w)
        sim.run()

    def test_move_along_forward_axis(self):
        sim = Simulator(gui=True, use_egl=False)
        scene = h5py.File('../data/scenes/200210_1654_manually_generated_scenes.hdf5')['scene'][0]
        sim.restore(scene, '../data/3d_models/shapenetsem40')
        sim.add_gripper('../simulator/gripper.urdf')
        p = [-.032, -.032, .1]
        z = [1, 1, 0]
        x = [1, -1, 0]
        w = .2
        sim.add_debug_pose(p, z, x, w)
        sim.run(epochs=100)
        sim.teleport_to_pre_grasp(p, z, x, w)

        sim.grasp_along(z)
        sim.move_to_post_grasp()

