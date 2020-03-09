from core.orthographic import OrthoNet
from simulator.simulator import Simulator

import os
import time
import h5py
import argparse
import logging
import datetime
import numpy as np


def transform_camera_to_world(cloud, camera):
    R = get_camera_frame(camera)
    t = np.array(camera.pos)[:, np.newaxis]
    return (R.dot(cloud.T) + t).T


def transform_world_to_camera(cloud, camera):
    R = get_camera_frame(camera).T
    t = R.dot(-np.array(camera.pos))[:, np.newaxis]
    return (R.dot(cloud.T) + t).T


def get_camera_frame(camera):
    """
    Computes the camera rotation matrix w.r.t world coordinates from a given camera instance
    :param camera: Sim camera instance
    :return: Transformation matrix representing the position and orientation of the camera
    """

    cW = np.array(camera.pos)  # Camera position wrt world
    tW = np.array(camera.target)  # Camera target wrt world

    # Basis of camera frame wrt world
    zW = tW - cW
    zW /= np.linalg.norm(zW)
    if np.linalg.norm(zW.flatten()[:2]) < 0.001:
        # If z is almost vertical, x is aligned with world's x
        xW = np.array([1, 0, 0])
    else:
        # Otherwise x is in the XY plane and orthogonal to z
        xW = np.array([zW[1], -zW[0], 0])
        xW = xW / np.linalg.norm(xW)
    # Right handed frame, y is computed from the other known axes
    yW = np.cross(zW, xW)

    return np.column_stack((xW, yW, zW))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network',
                        default='/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5',
                        type=str,
                        help='path to hdf5 file containing the network model')
    parser.add_argument('--scenes',
                        default='../data/scenes/shapenetsem40_5.hdf5',
                        type=str,
                        help='path to hdf5 file containing the simulation scenes')
    parser.add_argument('--angle',
                        default=45,
                        type=float,
                        help='angle in degrees in the y axes at which the camera will be placed')
    parser.add_argument('--distance',
                        default=1.5,
                        type=float,
                        help='distance in meters from the origin at which the camera will be placed')
    parser.add_argument('--output-path',
                        default=os.environ['RESULTS_PATH'],
                        type=str,
                        help='path to output directory where the results will be saved')
    parser.add_argument('--cam-resolution',
                        # TODO: replace this for a proper resolution once done testing
                        default=300,
                        type=int,
                        help='Resolution of the simulation camera. Relevant for the point cloud resolution')
    parser.add_argument('--output-file',
                        default='',
                        type=str,
                        help='Name of the output file. A date will be prepended')
    parser.add_argument('--padding',
                        default=80,
                        type=int,
                        help='Padding on the orthographic image sent to the network')
    parser.add_argument('--gui',
                        default=0,
                        type=int,
                        choices=[0, 1],
                        help='If set to 1 pybullet gui will be launched')
    parser.add_argument('--debug',
                        default=0,
                        type=int,
                        choices=[0, 1],
                        help='If set to 1 debug visualizations will be used in the orthographic pipeline and simulator')
    parser.add_argument('--omit-results',
                        default=1,
                        type=int,
                        choices=[0, 1],
                        help='If set to 1 it won\'t save any results. Useful for testing')

    args = parser.parse_args()

    FMT = "[%(levelname)s] [%(asctime)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FMT)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    results_fn = os.path.join(args.output_path, 'metric_baseline_%s_%s.txt' % (dt, args.output_file))
    if not args.omit_results:
        results_f = open(results_fn, 'w')
        results_f.writelines(['%s: %s\n' % (arg, getattr(args, arg)) for arg in vars(args)])
        results_f.write('scene_name,p,z,x,w,view,success\n')

    scenes_ds = h5py.File(args.scenes, 'r')
    scenes = scenes_ds['scene']

    # Uncomment to debug a particular scene
    # name_filter = scenes_ds['name'][:] == '1_2cc3904f7bfc8650ee25380b2e696b36'
    # scenes = scenes[name_filter]

    # Uncoment to debug a bunch of scenes by their name
    # name_filter = np.isin(scenes_ds['name'][:], ['3_2e228ee528f0a7054212ff51b27f0221', '0_1a4daa4904bb4a0949684e7f0bb99f9c', '1_1a4daa4904bb4a0949684e7f0bb99f9c', '2_1a4daa4904bb4a0949684e7f0bb99f9c', '3_1a4daa4904bb4a0949684e7f0bb99f9c', '4_1a4daa4904bb4a0949684e7f0bb99f9c', '1_2cc3904f7bfc8650ee25380b2e696b36', '0_1a0312faac503f7dc2c1a442b53fa053', '2_1a0312faac503f7dc2c1a442b53fa053', '4_1a0312faac503f7dc2c1a442b53fa053', '0_1f8a542e64756d349628684766da1bb4', '1_1f8a542e64756d349628684766da1bb4', '2_1f8a542e64756d349628684766da1bb4', '4_1f8a542e64756d349628684766da1bb4', '0_1f4e56064de606093e746e5f1700ce1a', '1_1f4e56064de606093e746e5f1700ce1a', '2_1f4e56064de606093e746e5f1700ce1a', '3_1f4e56064de606093e746e5f1700ce1a', '4_1f4e56064de606093e746e5f1700ce1a', '4_1d190c1bb38b29cb7a2fbdd8f7e098f4', '2_3ba7dd61736e7a96270c0e719fe4ed97', '0_1d4a469bdb53d3f77a3f900e0a6f2d83', '1_1d4a469bdb53d3f77a3f900e0a6f2d83', '2_1d4a469bdb53d3f77a3f900e0a6f2d83', '3_1d4a469bdb53d3f77a3f900e0a6f2d83', '4_1d4a469bdb53d3f77a3f900e0a6f2d83', '1_2d89d2b3b6749a9d99fbba385cc0d41d', '4_2d89d2b3b6749a9d99fbba385cc0d41d', '0_2f2f0e72a0088dd0f9b0754354ae88f5', '1_2f2f0e72a0088dd0f9b0754354ae88f5', '2_2f2f0e72a0088dd0f9b0754354ae88f5', '3_2f2f0e72a0088dd0f9b0754354ae88f5', '4_2f2f0e72a0088dd0f9b0754354ae88f5', '0_1be987c137d37f0b7c15f7bdb6fa82dd', '1_1be987c137d37f0b7c15f7bdb6fa82dd', '2_1be987c137d37f0b7c15f7bdb6fa82dd', '3_1be987c137d37f0b7c15f7bdb6fa82dd', '4_1be987c137d37f0b7c15f7bdb6fa82dd', '0_2daedbac8e1ee36f57467549cdfd9eb3', '2_2daedbac8e1ee36f57467549cdfd9eb3', '3_2daedbac8e1ee36f57467549cdfd9eb3', '4_2daedbac8e1ee36f57467549cdfd9eb3', '0_2c38b974e331ff14ec7d0aeaf786ab21', '2_2c38b974e331ff14ec7d0aeaf786ab21', '3_2c38b974e331ff14ec7d0aeaf786ab21', '4_2c38b974e331ff14ec7d0aeaf786ab21', '0_1e700065e92a072b39a22f83a4a90eb', '1_1e700065e92a072b39a22f83a4a90eb', '2_1e700065e92a072b39a22f83a4a90eb', '3_1e700065e92a072b39a22f83a4a90eb', '4_1e700065e92a072b39a22f83a4a90eb', '1_1cfc37465809382edfd1d17b67edb09', '3_1cfc37465809382edfd1d17b67edb09', '4_1cfc37465809382edfd1d17b67edb09', '1_1e227771ef66abdb4212ff51b27f0221', '1_1c5e5f1485ba5db1f879801ae14fa622', '2_1c5e5f1485ba5db1f879801ae14fa622', '3_1c5e5f1485ba5db1f879801ae14fa622', '4_1c5e5f1485ba5db1f879801ae14fa622', '4_1a0710af081df737c50a037462bade42', '2_1d9b04c979dfbddca84874d9f682ce6c', '3_1d9b04c979dfbddca84874d9f682ce6c', '4_1d9b04c979dfbddca84874d9f682ce6c', '0_3b947648cfb77c92dc6e31f308657eca', '1_3b947648cfb77c92dc6e31f308657eca', '2_3b947648cfb77c92dc6e31f308657eca', '3_3b947648cfb77c92dc6e31f308657eca', '4_3b947648cfb77c92dc6e31f308657eca', '3_1e23d88e517b711567ff608a5fbe6aa8'])
    # scenes = scenes[name_filter]

    # Uncomment to debug a range of scenes
    # scenes = scenes[4:9]

    # Uncomment to debug a particular scene by its index
    # scenes = [scenes[5]]

    # Hack to fix pybullet-pylab incompatibility on mac os
    if args.gui:
        import pylab as plt
        plt.figure()

    sim = Simulator(use_egl=False, gui=args.gui)  # Change to no gui
    sim.cam.pos = [0., np.cos(np.deg2rad(args.angle)) * args.distance, np.sin(np.deg2rad(args.angle)) * args.distance]
    sim.cam.width = args.cam_resolution
    sim.cam.height = args.cam_resolution
    sim.add_gripper(os.environ['GRIPPER_PATH'])

    onet = OrthoNet(model_fn=args.network)

    accuracy = 0

    _global_start = time.time()
    for scene_idx in range(len(scenes)):
        try:  # TODO: remove after fixing all bugs
            scene_name = scenes_ds['name'][scene_idx]
            logging.debug('Testing scene %s' % scene_name)
            sim.restore(scenes[scene_idx], os.environ['MODELS_PATH'])
            # Get the gripper out of the way so it doesn't interfere with cloud
            sim.teleport_to_pose([0., 0., 10.], [0., 0., 0.], 0.)

            logging.debug('Generating point cloud')
            _start = time.time()
            cloud = transform_world_to_camera(sim.cam.point_cloud(), sim.cam)
            _end = time.time()
            logging.debug('Done in %ss' % (_end - _start))

            logging.debug('Predicting')
            _start = time.time()
            ps, zs, xs, ws, scores, metadata = onet.predict(cloud,
                                                            onet.network_predictor,
                                                            predict_best_only=False,
                                                            n_attempts=5,
                                                            debug=args.debug,
                                                            padding=args.padding,
                                                            roi=[-2, 2, -2, 2, -0.01, 2],  # roi prevents opengl artifacts
                                                            )

            _end = time.time()
            logging.debug('Done in %ss' % (_end - _start))

            logging.debug('Evaluating')
            _start = time.time()

            result = False
            for view_idx in range(len(ps)):

                p = transform_camera_to_world(ps[view_idx], sim.cam)
                R = get_camera_frame(sim.cam)
                z = R.dot(zs[view_idx].T).T
                x = R.dot(xs[view_idx].T).T
                w = ws[view_idx]

                sim.add_debug_pose(p, z, x, w)
                sim.teleport_to_pre_grasp(p, z, x, w)
                sim.grasp_along(z)
                sim.move_to_post_grasp()
                result = result or sim.move_to_drop_off()
                if result:
                    break
                sim.restore(scenes[scene_idx], os.environ['MODELS_PATH'])

            _end = time.time()
            logging.debug('Evaluation %s and took %ss' % (['failed', 'succeeded'][result], _end - _start))

            if result:
                accuracy += 1

        except Exception, e:
            if not args.omit_results:
                results_f.write('%s failed due to exception: %s\n' % (scene_name, str(e)))
            logging.error(str(e))

    _global_end = time.time()
    if not args.omit_results:
        results_f.write('Total accuracy: %s' % (float(accuracy)/len(scenes)))
        results_f.write('Finished full evaluation in %s\n' % (_global_end - _global_start))
        results_f.close()
