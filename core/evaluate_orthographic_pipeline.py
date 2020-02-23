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
                        default='../data/scenes/200210_1654_manually_generated_scenes.hdf5',
                        type=str,
                        help='path to hdf5 file containing the simulation scenes')
    parser.add_argument('--angle',
                        default=90,
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

    args = parser.parse_args()

    FMT = "[%(asctime)s] %(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FMT)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    results_f = open(os.path.join(args.output_path, 'orthonet_%s.txt' % dt), 'w')
    results_f.writelines(['%s: %s\n' % (arg, getattr(args, arg)) for arg in vars(args)])
    results_f.write('scene_name,p,z,x,w,view,success\n')

    scenes_ds = h5py.File(args.scenes, 'r')
    scenes = scenes_ds['scene']  # TODO: remove selection

    sim = Simulator(use_egl=False, gui=False)  # Change to no gui
    sim.cam.pos = [0., np.cos(np.deg2rad(args.angle)) * args.distance, np.sin(np.deg2rad(args.angle)) * args.distance]
    sim.cam.width = args.cam_resolution
    sim.cam.height = args.cam_resolution
    sim.add_gripper(os.environ['GRIPPER_PATH'])

    onet = OrthoNet(model_fn=args.network)

    for scene_idx in range(len(scenes)):
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
                                                        predict_best_only=True,
                                                        n_attempts=5,
                                                        debug=False)
        _end = time.time()
        logging.debug('Done in %ss' % (_end - _start))
        best_idx = np.argmax(scores)

        p = transform_camera_to_world(ps[best_idx], sim.cam)
        R = get_camera_frame(sim.cam)
        z = R.dot(zs[best_idx].T).T
        x = R.dot(xs[best_idx].T).T
        w = ws[best_idx]

        sim.add_debug_pose(p, z, x, w)
        sim.teleport_to_pre_grasp(p, z, x, w)
        sim.grasp_along(z)
        sim.move_to_post_grasp()
        result = sim.move_to_drop_off()
        logging.debug('Evaluation %s' % (['failed', 'succeeded'][result]))
        # sim.run(1000)
        results_f.write(','.join([scene_name, str(p), str(z), str(x), str(w), metadata[best_idx]['view'], str(result)]) + '\n')
        results_f.flush()

    results_f.close()
