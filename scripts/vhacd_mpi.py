from utils import mpi
import os
import argparse
from mpi4py import MPI
import glob

class Job(mpi.Job):
    def __init__(self, in_fn, o_fn, res):
        self.in_fn = in_fn
        self.o_fn = o_fn
        self.res = res
        super(mpi.Job, self).__init__()

    def run(self):
        os.system('$VHACD_PATH --input {} --output {} --resolution {} > /dev/null'.format(self.in_fn, self.o_fn, self.res))

    @property
    def name(self):
        return self.in_fn

class Master(mpi.Master):
    def __init__(self, comm, path):
        self.obj_fns = glob.glob(path + '/*.obj')
        super(Master, self).__init__(comm)

    def setup(self):
        for obj_fn in self.obj_fns:
            out_fn = obj_fn.replace('.obj', '_vhacd.obj')
            self.jobs.append(Job(obj_fn, out_fn, 10000000))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to directory containing obj files')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master = Master(comm, args.path)
        master.run()
    else:
        slave = mpi.Slave(comm)
        slave.run()

