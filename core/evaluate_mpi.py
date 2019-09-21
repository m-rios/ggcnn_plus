import matplotlib
matplotlib.use('Agg')
import argparse
import glob
import os
import numpy as np
import network as net
from utils import mpi
import datetime
import h5py

from simulator.simulator import Simulator, VIDEO_LOGGER, STATE_LOGGER
from keras.models import load_model
from mpi4py import MPI

def print_attrs(scenes, f):
    f.write('SCENES ATTRS:\n')
    for item in scenes.attrs.items():
        f.write('{}: {}\n'.format(item[0], item[1]))
    f.write('---\n')

class JobEv(mpi.Job):
    def __init__(self, network, network_idx, depth, scene, scene_name, scene_idx, output_path, sim_log_path, args):
        self.net = network
        self.net_idx = network_idx
        self.depth = depth
        self.scene = scene
        self.scene_name = scene_name
        self.scene_idx = scene_idx
        self.output_path = output_path
        self.sim_log_path = sim_log_path
        self.result = None
        self.args = args
        super(JobEv, self).__init__()

    def run(self):
        sim = Simulator(timeout=4, use_egl=False, gui=self.args.gui)
        sim.cam.height = int(self.net.height)
        sim.cam.width = int(self.net.width)

        if self.args.subsample is not None:
            self.depth = net.subsample(self.depth, self.args.subsample)

        positions, angles, widths = self.net.predict(self.depth)

        sim.restore(self.scene, self.args.models)

        fn = os.path.join(self.output_path, self.scene_name +'.png')
        net.save_output_plot(self.depth, positions,
                angles, widths, fn, 1)

        gs = net.get_grasps_from_output(positions,
                angles, widths, n_grasps=1)

        if len(gs) > 0:
            gs = gs[0]

            pose, grasp_width = sim.cam.compute_grasp(gs.as_bb.points, self.depth[gs.center])
            pose = np.concatenate((pose, [0, 0, gs.angle]))

            if not self.args.nolog:
                if self.args.logvideo:
                    sim.start_log(self.sim_log_path + '/'+self.scene_name+'.mp4', VIDEO_LOGGER, rate=25)
                else:
                    sim.start_log(self.sim_log_path + '/'+self.scene_name+'.log', STATE_LOGGER)
            self.result = sim.evaluate_grasp(pose, grasp_width)
            if not self.args.nolog:
                sim.stop_log()
        else:
            self.result = False
        sim.disconnect()
        return self

    @property
    def name(self):
        return 'Epoch: {} Scene: {}'.format(self.net.epoch, self.scene_name)

class MasterEv(mpi.Master):
    def __init__(self, comm, data):
        self.results = None
        self.results_f = None
        self.progress = 0

        self.data = data
        self.model_fns = glob.glob(data.model + '/*.hdf5')
        self.model_name = self.model_fns[0].split('/')[-2]
        self.scenes = h5py.File(data.scenes, 'r')
        self.nscenes = self.scenes['name'].size
        self.dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
        self.model_results_path = os.path.join(data.results, '{}_{}'.format(self.dt, self.model_name))
        if not os.path.exists(self.model_results_path):
            os.makedirs(self.model_results_path)
        self.results_fn = self.model_results_path + '/results.txt'
        self.results_f = open(self.results_fn, 'w')
        self.results_f.write('ARGUMENTS:\n'+''.join(['{}: {}\n'.format(item[0], item[1]) for item in vars(data).items()]))
        self.results_f.write('---\n')
        print_attrs(self.scenes, self.results_f)

        self.results = np.zeros((len(self.model_fns), self.nscenes))

        _, height, width, _ = load_model(self.model_fns[0]).input_shape
        self.depth = self.scenes['depth'][:]
        super(MasterEv, self).__init__(comm, data)

    def setup(self):
        args = self.data

        for net_idx, model_fn in enumerate(self.model_fns):
            epoch = int(model_fn.split('_')[-2])

            if args.e is not None and epoch not in args.e:
                continue

            epoch_results_path = os.path.join(self.model_results_path, str(epoch))
            output_path = os.path.join(epoch_results_path, 'output')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            sim_log_path = os.path.join(epoch_results_path, 'sim_logs')
            if not os.path.exists(sim_log_path):
                os.makedirs(sim_log_path)

            network = net.Network(model_fn=model_fn)
            for scene_idx in range(self.scenes['name'].size):
                job = JobEv(network, net_idx, self.depth[scene_idx],
                        self.scenes['scene'][scene_idx], self.scenes['name'][scene_idx],
                        scene_idx, output_path, sim_log_path, args)
                self.jobs.append(job)

    def process_result(self, result):
        self.progress += 1
        print('Progress: {}%, result from {}: {}'.format(float(self.progress)/len(self.jobs), result.name, result.result))
        self.results[result.net_idx, result.scene_idx] = result.result
        if not result.result:
            epoch_results_path = os.path.join(self.model_results_path, str(result.net.epoch))
            sim_log_path = os.path.join(epoch_results_path, 'sim_logs')
            with open(os.path.join(sim_log_path, 'failures.txt'), 'a+') as f:
                scene_name = result.scene_name
                f.write(scene_name + '\n')

    def compile_results(self):
        accuracies = np.sum(self.results, axis=1)/self.results.shape[1]
        for net_idx, acc in enumerate(accuracies):
            epoch = int(self.model_fns[net_idx].split('_')[-2])
            self.results_f.write('Epoch {}: {}%\n'.format(epoch, acc))
        self.results_f.close()

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument('model', help='path to the root directory of a model')
        parser.add_argument('--grasps', default=1, help='Number of grasps predicted per image')
        parser.add_argument('--results', default=os.environ['RESULTS_PATH'], help='Path to simulation log files')
        parser.add_argument('--scenes', default=os.environ['GGCNN_SCENES_PATH'], help='Path to scene files location')
        parser.add_argument('--models', default=os.environ['MODELS_PATH'], help='Path to obj files location')
        parser.add_argument('--logvideo', action='store_true')
        parser.add_argument('--nolog', action='store_true')
        parser.add_argument('--gui', action='store_true')
        parser.add_argument('--subsample', default=None, type=float, help='Subsample depth image by provided factor before feeding to network')
        parser.add_argument('-e', nargs='+', default=None, type=int, help='epochs to evaluate, if next arg is model, separate with -- ')

        args = parser.parse_args()

        master = MasterEv(comm, args)
        master.run()
    else:
        slave = mpi.Slave(comm)
        slave.run()
