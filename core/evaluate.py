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
import datetime
import h5py
import numpy as np

from simulator.simulator import Simulator, VIDEO_LOGGER, STATE_LOGGER
import network as net

def print_attrs(scenes, f):
    f.write('SCENES ATTRS:\n')
    for item in scenes.attrs.items():
        f.write('{}: {}\n'.format(item[0], item[1]))
    f.write('---\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to the root directory of a model')
    parser.add_argument('--grasps', default=1, help='Number of grasps predicted per image')
    parser.add_argument('--results', default=os.environ['RESULTS_PATH'], help='Path to simulation log files')
    parser.add_argument('--scenes', default=os.environ['GGCNN_SCENES_PATH'], help='Path to scene files location')
    parser.add_argument('--models', default=os.environ['MODELS_PATH'], help='Path to obj files location')
    parser.add_argument('--logvideo', action='store_true')
    parser.add_argument('--subsample', default=None, type=float, help='Subsample depth image by provided factor before feeding to network')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('-e', nargs='+', default=None, type=int, help='epochs to evaluate, if next arg is model, separate with -- ')

    args = parser.parse_args()

    model_fns = glob.glob(args.model + '/*.hdf5')
    assert len(model_fns) > 0
    model_name = model_fns[0].split('/')[-2]

    # Get input size and initialize simulator camera with it
    from keras.models import load_model
    _, height, width, _ = load_model(model_fns[0]).input_shape

    sim = Simulator(gui=args.gui, timeout=4, debug=True, use_egl=False)
    sim.cam.height = height
    sim.cam.width = width

    scenes = h5py.File(args.scenes, 'r')

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    model_results_path = os.path.join(args.results, '{}_{}'.format(dt, model_name))
    if not os.path.exists(model_results_path):
        os.makedirs(model_results_path)
    results_fn = model_results_path + '/results.txt'
    results_f = open(results_fn, 'w')
    results_f.write('ARGUMENTS:\n'+''.join(['{}: {}\n'.format(item[0], item[1]) for item in vars(args).items()]))
    results_f.write('---\n')
    print_attrs(scenes, results_f)

    depth = scenes['depth'][:]

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
        for scene_idx in range(scenes['name'].size):
            scene_name = scenes['name'][scene_idx]
            sim.restore(scenes['scene'][scene_idx], args.models)

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

                if args.logvideo:
                    sim.start_log(sim_log_path + '/'+scene_name+'.mp4', VIDEO_LOGGER, rate=25)
                else:
                    sim.start_log(sim_log_path + '/'+scene_name+'.log', STATE_LOGGER)
                result = sim.evaluate_grasp(pose, grasp_width)
                sim.stop_log()
            else:
                result = False

            if not result:
                failures.write(scene_name + '\n')
            results += result

        success = float(results)/float(scenes['name'].size)
        results_f.write('Epoch {}: {}%\n'.format(epoch, success))
        results_f.flush()
        failures.close()


