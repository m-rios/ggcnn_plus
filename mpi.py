"""
Inspired by https://gist.github.com/fspaolo/51eaf5a20d6d418bd4d0
"""
from mpi4py import MPI
from abc import ABCMeta, abstractmethod

DIETAG = 0
WORKTAG = 1

class Job(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def run(self):
        pass

    @property
    def name(self):
        return ''

class Master():
    __metaclass__ = ABCMeta
    def __init__(self, comm, data=None):
        self.comm = comm
        self.status = MPI.Status()
        self.nworkers = comm.Get_size()
        self.jobs = []
        self.data = data
        self.setup()
        super(Master, self).__init__()

    @abstractmethod
    def setup(self):
        raise NotImplemented

    def process_result(self, result):
        pass

    def compile_results(self):
        pass

    def run(self):
        # Send first batch of jobs to workers
        for rank in range(1, self.nworkers):
            job = self.jobs[rank-1]
            self.comm.send(job, dest=rank, tag=WORKTAG)
        # Keep sending remaining jobs
        for job in self.jobs[self.nworkers:]:
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            self.process_result(result)
            dest = self.status.Get_source()
            print('Sending job {} to {}'.format(job.name, dest))
            self.comm.send(job, dest=dest, tag=WORKTAG)
        # Wait for last jobs to finish
        for rank in range(1, self.nworkers):
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            self.process_result(result)
        # Kill slaves
        for rank in range(1, self.nworkers):
            self.comm.send(0, dest=rank, tag=DIETAG)

        self.compile_results()


class Slave(object):
    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.status = MPI.Status()

    def run(self):
        while True:
            job = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
            if self.status.Get_tag() == DIETAG:
                break
            result = job.run()
            self.comm.send(result, dest=0, tag=0)

if __name__ == '__main__':
    import random
    from time import sleep
    class JobTest(Job):
        def run(self):
            sleep(0.5)
            return random.getrandbits(1)

    class MasterTest(Master):
        def __init__(self, comm):
            self.results = []
            super(MasterTest, self).__init__(comm)

        def setup(self):
            for _ in range(100):
                self.jobs.append(JobTest())

        def process_result(self, result):
            self.results.append(result)

        def compile_results(self):
            result = float(sum(self.results))/len(self.results)
            print('Result: {}%'.format(result))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master = MasterTest(comm)
        master.run()
    else:
        slave = Slave(comm)
        slave.run()

