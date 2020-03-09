from core.network import Network, get_grasps_from_output, plot_grasps
from simulator.simulator import Simulator
from scipy.spatial.transform import Rotation as R

import os
import time
import h5py
import argparse
import logging
import datetime
import numpy as np


def get_pose(depth, positions, angles, widths, cam):
    z = np.array(cam.target) - np.array(cam.pos)
    z = z / np.linalg.norm(z)

    gs = get_grasps_from_output(positions, angles, widths, n_grasps=5)

    if len(gs) == 0:
        return None

    for g_idx, g in enumerate(gs):
        d = depth[g.center]
        p, w = cam.compute_grasp(g.as_bb.points, d)

        if np.linalg.norm(z.flatten()[:2]) < 0.001:
            # If z is almost vertical, x is aligned with world's x
            x = np.array([1, 0, 0])
        else:
            # Otherwise x is in the XY plane and orthogonal to z
            x = np.array([z[1], -z[0], 0])
            x = x / np.linalg.norm(x)

        print 'Angle', g.angle
        r = R.from_rotvec(z * g.angle)
        x = r.apply(x)

        start = p - (2 * x / w)
        end = p + (2 * x / w)
        if start[2] > 0 or end[2] > 0:
            logging.debug('Found valid grasp in attempt %s/%s' % (g_idx + 1, len(gs)))
            break
    else:
        logging.debug('Valid grasp not found after %s attempts' % (len(gs)))

    return p, z, x, w


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
                        default=60,
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
    parser.add_argument('--output-file',
                        default='',
                        type=str,
                        help='Name of the output file. A date will be prepended')
    parser.add_argument('--save-grasps',
                        action='store_true',
                        help='If set it will save the output of the network to an image')
    parser.add_argument('--gui',
                        default=1,
                        type=int,
                        choices=[0, 1],
                        help='If set to 1 pybullet gui will be launched')
    parser.add_argument('--debug',
                        default=1,
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
    results_fn = os.path.join(args.output_path, 'straight_%s_%s.txt' % (dt, args.output_file))
    if not args.omit_results:
        results_f = open(results_fn, 'w')
        results_f.writelines(['%s: %s\n' % (arg, getattr(args, arg)) for arg in vars(args)])
        results_f.write('scene_name,p,z,x,w,view,success\n')

    # if args.save_grasps:
    #     grasps_path = results_fn.split('.txt')[0]
    #     if not os.path.exists(grasps_path):
    #         os.makedirs(grasps_path)

    scenes_ds = h5py.File(args.scenes, 'r')
    scenes = scenes_ds['scene']

    # Uncomment to debug a particular scene
    # name_filter = scenes_ds['name'][:] == '0_1a4daa4904bb4a0949684e7f0bb99f9c'
    # scenes = scenes[name_filter]

    name_filter = np.isin(scenes_ds['name'][:], ['0_2e228ee528f0a7054212ff51b27f0221','1_2e228ee528f0a7054212ff51b27f0221','2_2e228ee528f0a7054212ff51b27f0221','2_1a4daa4904bb4a0949684e7f0bb99f9c','4_1a4daa4904bb4a0949684e7f0bb99f9c','0_2cc3904f7bfc8650ee25380b2e696b36','1_2cc3904f7bfc8650ee25380b2e696b36','2_2cc3904f7bfc8650ee25380b2e696b36','3_2cc3904f7bfc8650ee25380b2e696b36','4_2cc3904f7bfc8650ee25380b2e696b36','1_2ed76efe98e1d6a8e4670226b51cdc8','0_1cc93f96ad5e16a85d3f270c1c35f1c7','1_1cc93f96ad5e16a85d3f270c1c35f1c7','2_1cc93f96ad5e16a85d3f270c1c35f1c7','4_1cc93f96ad5e16a85d3f270c1c35f1c7','0_3b1f7f066991f2d45969e7cd6a0b6a55','1_3b1f7f066991f2d45969e7cd6a0b6a55','2_3b1f7f066991f2d45969e7cd6a0b6a55','3_3b1f7f066991f2d45969e7cd6a0b6a55','4_3b1f7f066991f2d45969e7cd6a0b6a55','0_1a0312faac503f7dc2c1a442b53fa053','1_1a0312faac503f7dc2c1a442b53fa053','2_1a0312faac503f7dc2c1a442b53fa053','2_3b4cbd4fd5f6819bea4732296ea50647','3_3b4cbd4fd5f6819bea4732296ea50647','4_3b4cbd4fd5f6819bea4732296ea50647','0_1bf3d5cc59b63cd6e979876000467c9c','2_1bf3d5cc59b63cd6e979876000467c9c','3_1bf3d5cc59b63cd6e979876000467c9c','4_1bf3d5cc59b63cd6e979876000467c9c','0_1b36df7ed7ddd974c538fbfc1e40dbe0','3_1b36df7ed7ddd974c538fbfc1e40dbe0','4_1b36df7ed7ddd974c538fbfc1e40dbe0','0_1f8a542e64756d349628684766da1bb4','1_1f8a542e64756d349628684766da1bb4','2_1f8a542e64756d349628684766da1bb4','3_1f8a542e64756d349628684766da1bb4','4_1f8a542e64756d349628684766da1bb4','0_2c2f99c7fc9e914d978eac5bf3137468','2_2c2f99c7fc9e914d978eac5bf3137468','3_2c2f99c7fc9e914d978eac5bf3137468','0_1ec297183c8aa37a36c7d12bccd8bbd','1_1ec297183c8aa37a36c7d12bccd8bbd','2_1ec297183c8aa37a36c7d12bccd8bbd','3_1ec297183c8aa37a36c7d12bccd8bbd','4_1ec297183c8aa37a36c7d12bccd8bbd','0_3a7840f2b310d62294a9d0491b6eccf9','1_3a7840f2b310d62294a9d0491b6eccf9','3_3a7840f2b310d62294a9d0491b6eccf9','4_3a7840f2b310d62294a9d0491b6eccf9','0_2bfb9b5ce81f5388ed311a82ec98a0c2','4_2bfb9b5ce81f5388ed311a82ec98a0c2','0_1f4e56064de606093e746e5f1700ce1a','1_1f4e56064de606093e746e5f1700ce1a','2_1f4e56064de606093e746e5f1700ce1a','4_1f4e56064de606093e746e5f1700ce1a','0_2ba1891e2b354b219617cbc6647fc553','3_2ba1891e2b354b219617cbc6647fc553','4_2ba1891e2b354b219617cbc6647fc553','0_1d190c1bb38b29cb7a2fbdd8f7e098f4','1_1d190c1bb38b29cb7a2fbdd8f7e098f4','2_1d190c1bb38b29cb7a2fbdd8f7e098f4','3_1d190c1bb38b29cb7a2fbdd8f7e098f4','0_3ba7dd61736e7a96270c0e719fe4ed97','2_3ba7dd61736e7a96270c0e719fe4ed97','3_3ba7dd61736e7a96270c0e719fe4ed97','0_1d4a469bdb53d3f77a3f900e0a6f2d83','1_1d4a469bdb53d3f77a3f900e0a6f2d83','2_1d4a469bdb53d3f77a3f900e0a6f2d83','3_1d4a469bdb53d3f77a3f900e0a6f2d83','0_2d89d2b3b6749a9d99fbba385cc0d41d','1_2d89d2b3b6749a9d99fbba385cc0d41d','2_2d89d2b3b6749a9d99fbba385cc0d41d','4_2d89d2b3b6749a9d99fbba385cc0d41d','1_2f2f0e72a0088dd0f9b0754354ae88f5','2_2f2f0e72a0088dd0f9b0754354ae88f5','4_2f2f0e72a0088dd0f9b0754354ae88f5','2_1be987c137d37f0b7c15f7bdb6fa82dd','4_1be987c137d37f0b7c15f7bdb6fa82dd','0_3a5351666689a7b2b788559e93c74a0f','1_3a5351666689a7b2b788559e93c74a0f','2_3a5351666689a7b2b788559e93c74a0f','3_3a5351666689a7b2b788559e93c74a0f','0_2daedbac8e1ee36f57467549cdfd9eb3','1_2daedbac8e1ee36f57467549cdfd9eb3','2_2daedbac8e1ee36f57467549cdfd9eb3','4_2daedbac8e1ee36f57467549cdfd9eb3','1_1be58678b919b12bc5fe7f65b41f3b19','3_1be58678b919b12bc5fe7f65b41f3b19','0_2c38b974e331ff14ec7d0aeaf786ab21','2_2c38b974e331ff14ec7d0aeaf786ab21','3_2c38b974e331ff14ec7d0aeaf786ab21','4_2c38b974e331ff14ec7d0aeaf786ab21','2_2f55f20282971f7125c70fb1df3f879b','4_2f55f20282971f7125c70fb1df3f879b','1_1e700065e92a072b39a22f83a4a90eb','0_1cfc37465809382edfd1d17b67edb09','1_1cfc37465809382edfd1d17b67edb09','4_1cfc37465809382edfd1d17b67edb09','1_2b28e2a5080101d245af43a64155c221','3_2b28e2a5080101d245af43a64155c221','4_2b28e2a5080101d245af43a64155c221','0_1e227771ef66abdb4212ff51b27f0221','2_1e227771ef66abdb4212ff51b27f0221','3_1e227771ef66abdb4212ff51b27f0221','4_1e227771ef66abdb4212ff51b27f0221','0_1c5e5f1485ba5db1f879801ae14fa622','1_1c5e5f1485ba5db1f879801ae14fa622','2_1c5e5f1485ba5db1f879801ae14fa622','4_1c5e5f1485ba5db1f879801ae14fa622','0_1a0710af081df737c50a037462bade42','1_1a0710af081df737c50a037462bade42','3_1a0710af081df737c50a037462bade42','0_1d4480abe9aa45ce51a99c0e19a8a54','1_1d4480abe9aa45ce51a99c0e19a8a54','2_1d4480abe9aa45ce51a99c0e19a8a54','3_1d4480abe9aa45ce51a99c0e19a8a54','2_1d9b04c979dfbddca84874d9f682ce6c','4_1d9b04c979dfbddca84874d9f682ce6c','1_1b370d5326cb7da75318625c74026d6','2_1b370d5326cb7da75318625c74026d6','0_3b947648cfb77c92dc6e31f308657eca','1_3b947648cfb77c92dc6e31f308657eca','2_3b947648cfb77c92dc6e31f308657eca','4_3b947648cfb77c92dc6e31f308657eca','0_1e23d88e517b711567ff608a5fbe6aa8','2_1e23d88e517b711567ff608a5fbe6aa8'])
    # name_filter = np.isin(scenes_ds['name'][:],['0_2e228ee528f0a7054212ff51b27f0221', '2_2e228ee528f0a7054212ff51b27f0221', '0_1a4daa4904bb4a0949684e7f0bb99f9c',
    #  '1_1a4daa4904bb4a0949684e7f0bb99f9c', '1_2cc3904f7bfc8650ee25380b2e696b36', '3_2cc3904f7bfc8650ee25380b2e696b36',
    #  '4_2cc3904f7bfc8650ee25380b2e696b36', '1_2ed76efe98e1d6a8e4670226b51cdc8', '2_2ed76efe98e1d6a8e4670226b51cdc8',
    #  '3_2ed76efe98e1d6a8e4670226b51cdc8', '1_1cc93f96ad5e16a85d3f270c1c35f1c7', '2_1cc93f96ad5e16a85d3f270c1c35f1c7',
    #  '3_1cc93f96ad5e16a85d3f270c1c35f1c7', '4_1cc93f96ad5e16a85d3f270c1c35f1c7', '0_3b1f7f066991f2d45969e7cd6a0b6a55',
    #  '2_3b1f7f066991f2d45969e7cd6a0b6a55', '3_3b1f7f066991f2d45969e7cd6a0b6a55', '4_3b1f7f066991f2d45969e7cd6a0b6a55',
    #  '2_1a0312faac503f7dc2c1a442b53fa053', '3_1a0312faac503f7dc2c1a442b53fa053', '0_3b4cbd4fd5f6819bea4732296ea50647',
    #  '1_3b4cbd4fd5f6819bea4732296ea50647', '2_3b4cbd4fd5f6819bea4732296ea50647', '3_3b4cbd4fd5f6819bea4732296ea50647',
    #  '4_3b4cbd4fd5f6819bea4732296ea50647', '0_1bf3d5cc59b63cd6e979876000467c9c', '1_1bf3d5cc59b63cd6e979876000467c9c',
    #  '3_1bf3d5cc59b63cd6e979876000467c9c', '4_1bf3d5cc59b63cd6e979876000467c9c', '0_1b36df7ed7ddd974c538fbfc1e40dbe0',
    #  '1_1b36df7ed7ddd974c538fbfc1e40dbe0', '2_1b36df7ed7ddd974c538fbfc1e40dbe0', '3_1b36df7ed7ddd974c538fbfc1e40dbe0',
    #  '4_1b36df7ed7ddd974c538fbfc1e40dbe0', '0_1f8a542e64756d349628684766da1bb4', '1_1f8a542e64756d349628684766da1bb4',
    #  '2_1f8a542e64756d349628684766da1bb4', '4_1f8a542e64756d349628684766da1bb4', '0_2c2f99c7fc9e914d978eac5bf3137468',
    #  '0_1ec297183c8aa37a36c7d12bccd8bbd', '3_1ec297183c8aa37a36c7d12bccd8bbd', '2_3a7840f2b310d62294a9d0491b6eccf9',
    #  '1_2bfb9b5ce81f5388ed311a82ec98a0c2', '0_1f4e56064de606093e746e5f1700ce1a', '3_1f4e56064de606093e746e5f1700ce1a',
    #  '4_1f4e56064de606093e746e5f1700ce1a', '0_2ba1891e2b354b219617cbc6647fc553', '4_2ba1891e2b354b219617cbc6647fc553',
    #  '0_1d190c1bb38b29cb7a2fbdd8f7e098f4', '4_1d190c1bb38b29cb7a2fbdd8f7e098f4', '1_3ba7dd61736e7a96270c0e719fe4ed97',
    #  '2_3ba7dd61736e7a96270c0e719fe4ed97', '3_3ba7dd61736e7a96270c0e719fe4ed97', '0_1d4a469bdb53d3f77a3f900e0a6f2d83',
    #  '1_1d4a469bdb53d3f77a3f900e0a6f2d83', '1_1b64b36bf7ddae3d7ad11050da24bb12', '2_1b64b36bf7ddae3d7ad11050da24bb12',
    #  '3_1b64b36bf7ddae3d7ad11050da24bb12', '1_2d89d2b3b6749a9d99fbba385cc0d41d', '2_2d89d2b3b6749a9d99fbba385cc0d41d',
    #  '3_2d89d2b3b6749a9d99fbba385cc0d41d', '4_2d89d2b3b6749a9d99fbba385cc0d41d', '0_2f2f0e72a0088dd0f9b0754354ae88f5',
    #  '1_2f2f0e72a0088dd0f9b0754354ae88f5', '2_2f2f0e72a0088dd0f9b0754354ae88f5', '3_2f2f0e72a0088dd0f9b0754354ae88f5',
    #  '4_2f2f0e72a0088dd0f9b0754354ae88f5', '0_1be987c137d37f0b7c15f7bdb6fa82dd', '2_1be987c137d37f0b7c15f7bdb6fa82dd',
    #  '4_1be987c137d37f0b7c15f7bdb6fa82dd', '1_3a5351666689a7b2b788559e93c74a0f', '2_3a5351666689a7b2b788559e93c74a0f',
    #  '3_3a5351666689a7b2b788559e93c74a0f', '4_3a5351666689a7b2b788559e93c74a0f', '0_2daedbac8e1ee36f57467549cdfd9eb3',
    #  '4_2daedbac8e1ee36f57467549cdfd9eb3', '0_1be58678b919b12bc5fe7f65b41f3b19', '2_1be58678b919b12bc5fe7f65b41f3b19',
    #  '3_1be58678b919b12bc5fe7f65b41f3b19', '4_1be58678b919b12bc5fe7f65b41f3b19', '3_2c38b974e331ff14ec7d0aeaf786ab21',
    #  '4_2c38b974e331ff14ec7d0aeaf786ab21', '0_2f55f20282971f7125c70fb1df3f879b', '1_2f55f20282971f7125c70fb1df3f879b',
    #  '2_2f55f20282971f7125c70fb1df3f879b', '3_2f55f20282971f7125c70fb1df3f879b', '0_1e700065e92a072b39a22f83a4a90eb',
    #  '1_1e700065e92a072b39a22f83a4a90eb', '2_1e700065e92a072b39a22f83a4a90eb', '3_1e700065e92a072b39a22f83a4a90eb',
    #  '4_1e700065e92a072b39a22f83a4a90eb', '2_1cfc37465809382edfd1d17b67edb09', '3_1cfc37465809382edfd1d17b67edb09',
    #  '4_1cfc37465809382edfd1d17b67edb09', '0_2b28e2a5080101d245af43a64155c221', '2_2b28e2a5080101d245af43a64155c221',
    #  '3_2b28e2a5080101d245af43a64155c221', '4_2b28e2a5080101d245af43a64155c221', '0_1e227771ef66abdb4212ff51b27f0221',
    #  '2_1c5e5f1485ba5db1f879801ae14fa622', '4_1c5e5f1485ba5db1f879801ae14fa622', '1_1a0710af081df737c50a037462bade42',
    #  '2_1a0710af081df737c50a037462bade42', '3_1a0710af081df737c50a037462bade42', '4_1a0710af081df737c50a037462bade42',
    #  '0_1d4480abe9aa45ce51a99c0e19a8a54', '1_1d4480abe9aa45ce51a99c0e19a8a54', '2_1d4480abe9aa45ce51a99c0e19a8a54',
    #  '3_1d4480abe9aa45ce51a99c0e19a8a54', '4_1d4480abe9aa45ce51a99c0e19a8a54', '1_1d9b04c979dfbddca84874d9f682ce6c',
    #  '2_1d9b04c979dfbddca84874d9f682ce6c', '4_1d9b04c979dfbddca84874d9f682ce6c', '0_1b370d5326cb7da75318625c74026d6',
    #  '1_1b370d5326cb7da75318625c74026d6', '3_1b370d5326cb7da75318625c74026d6', '4_1b370d5326cb7da75318625c74026d6',
    #  '0_3b947648cfb77c92dc6e31f308657eca', '3_3b947648cfb77c92dc6e31f308657eca', '1_1e23d88e517b711567ff608a5fbe6aa8'])
    scenes = scenes[name_filter]

    # Hack to fix pybullet-pylab incompatibility on mac os
    if args.gui:
        import pylab as plt
        plt.figure()

    if args.debug:
        plt.ion()

    sim = Simulator(use_egl=False, gui=args.gui)  # Change to no gui
    sim.cam.pos = [0., np.cos(np.deg2rad(args.angle)) * args.distance, np.sin(np.deg2rad(args.angle)) * args.distance]
    sim.add_gripper(os.environ['GRIPPER_PATH'])

    net = Network(model_fn=args.network)

    _global_start = time.time()
    for scene_idx in range(len(scenes)):
        try:
            scene_name = scenes_ds['name'][scene_idx]
            logging.debug('Testing scene %s' % scene_name)
            sim.restore(scenes[scene_idx], os.environ['MODELS_PATH'])
            # Get the gripper out of the way so it doesn't interfere with camera
            sim.teleport_to_pose([0., 0., 10.], [0., 0., 0.], 0.)

            _, depth = sim.cam.snap()

            logging.debug('Predicting')
            _start = time.time()

            pos, ang, wid = net.predict(depth)

            _end = time.time()
            logging.debug('Done in %ss' % (_end - _start))

            if args.debug:
                plt.close()
                plot_grasps(depth, get_grasps_from_output(pos, ang, wid, n_grasps=5))
                plt.pause(0.001)

            pose = get_pose(depth, pos, ang, wid, sim.cam)

            if pose is None:
                results_f.write(','.join([scene_name, 'N/A', 'N/A', 'N/A', 'N/A', False]) + '\n')
                continue
            else:
                p, z, x, w = pose

            logging.debug('Evaluating')
            _start = time.time()
            sim.add_debug_pose(p, z, x, w)
            sim.teleport_to_pre_grasp(p, z, x, w)
            sim.grasp_along(z)
            sim.move_to_post_grasp()
            result = sim.move_to_drop_off()
            _end = time.time()
            logging.debug('Evaluation %s and took %ss' % (['failed', 'succeeded'][result], _end - _start))
            if not args.omit_results:
                results_f.write(','.join([scene_name, str(p), str(z), str(x), str(w), str(result)]) + '\n')
                results_f.flush()
        except TypeError, e:
            if not args.omit_results:
                results_f.write('%s failed due to exception: %s\n' % (scene_name, str(e)))
            logging.error(str(e))

    _global_end = time.time()
    if not args.omit_results:
        results_f.write('Finished full evaluation in %s\n' % (_global_end - _global_start))
        results_f.close()
