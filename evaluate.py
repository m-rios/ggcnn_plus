"""
Evaluation output:
data:
    evaluation:
        model_name:
            epoch:
                output:
                    output_imgs
                sim_logs:
                    sim_logs
        results.txt
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import gaussian
from ggcnn.dataset_processing.grasp import detect_grasps, BoundingBoxes
from simulator.simulator import Simulator
import network as net

SCENES_PATH = os.environ['GGCNN_SCENES_PATH']
SHAPENET_PATH = os.environ['SHAPENET_PATH']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the root directory of a model')
    parser.add_argument('--grasps', default=1, help='Number of grasps predicted per image')
    parser.add_argument('--results_path', default=os.environ['RESULTS_PATH'], help='Path to simulation log files')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('-e', nargs='+', default=None, type=int, help='epochs to evaluate, if next arg is model, separate with -- ')

    args = parser.parse_args()

    model_fns = glob.glob(args.model + '/*.hdf5')
    assert len(model_fns) > 0
    model_name = model_fns[0].split('/')[-2]

    # Get input size and initialize simulator camera with it
    from keras.models import load_model
    _, height, width, _ = load_model(model_fns[0]).input_shape

    sim = Simulator(gui=args.gui, timeout=4, debug=True)
    sim.cam.height = height
    sim.cam.width = width

    scene_fns = glob.glob(SCENES_PATH + '/*.csv')

    model_results_path = os.path.join(args.results_path, model_name)
    if not os.path.exists(model_results_path):
        os.makedirs(model_results_path)
    results_fn = model_results_path + '/results.txt'
    results_f = open(results_fn, 'w')

    rgb, depth = net.read_input_from_scenes(scene_fns, width, height)

    # Iterate through epochs
    for model_fn in model_fns:
        epoch = int(model_fn.split('_')[-2])

        if args.e is not None and epoch not in args.e:
            continue

        epoch_results_path = os.path.join(model_results_path, str(epoch))
        output_path = os.path.join(epoch_results_path, 'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        sim_log_path = os.path.join(epoch_results_path, 'sim_logs')
        if not os.path.exists(sim_log_path):
            os.makedirs(sim_log_path)


        print('Evaluating epoch {} model {}'.format(model_name, epoch))

        network = net.Network(model_fn)

        positions, angles, widths = network.predict(depth)

        results = 0
        failures = open(os.path.join(sim_log_path, 'failures.txt'), 'w')

        # Test results for each scene
        for scene_idx, scene_fn in enumerate(scene_fns):
            scene_name = scene_fn.split('/')[-1].split('.')[-2].split('_')[-2]
            sim.restore(scene_fn, SHAPENET_PATH)

            if not args.gui:
                fn = os.path.join(output_path, scene_name +'.png')
                net.save_output_plot(depth[scene_idx], positions[scene_idx],
                        angles[scene_idx], widths[scene_idx], fn, args.grasps)

            # Compute grasp 6DOF coordiantes w.r.t camera frame
            gs = net.get_grasps_from_output(positions[scene_idx],
                    angles[scene_idx], widths[scene_idx], n_grasps=args.grasps)
            if len(gs) > 0:
                gs = gs[0]

                pose, grasp_width = sim.cam.compute_grasp(gs.as_bb.points, depth[scene_idx][gs.center])
                pose = np.concatenate((pose, [0, 0, gs.angle]))

                result = sim.evaluate_grasp(pose, grasp_width, sim_log_path + '/'+scene_name+'.log')
            else:
                result = False

            if not result:
                failures.write(scene_name + '\n')
            results += result

        success = float(results)/float(len(scene_fns))
        results_f.write('Epoch {}: {}%\n'.format(epoch, success))
        failures.close()


