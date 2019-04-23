import pandas as pd
import glob
import os
import sys
from subprocess import Popen
import numpy as np
from random import shuffle
from contextlib import contextmanager

VHACD_PATH = os.environ['VHACD_PATH']
VHACD_LOG_FILE = '/tmp/vhacd.log'

class VHACD:
    def __init__(self):

        self.vhacd_path = VHACD_PATH
        # Maximum number of voxels generated during the voxelization stage
        self.resolution = 100000
        # Maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane
        self.depth = 20
        # Maximum concavity
        self.concavity = 0.0025
        self.concavity = 0.0
        # Granularity of the search for the "best" clipping plane
        self.planeDownsampling = 4
        # Precision of the convex-hull generation process during the clipping plane selection stage
        self.convexhullDownsampling = 4
        # Bias toward clipping along symmetry planes
        self.alpha = 0.05
        # Bias toward clipping along revolution axes'
        self.beta = 0.05
        # Maximum allowed concavity during the merge stage
        self.gamma = 0.05
        self.gamma = 0.0
        # Enable/disable normalizing the mesh before applying the convex decomposition
        self.pca = False
        # Approximate convex decomposition mode
        self.mode = 'VOXEL'
        # Maximum number of vertices per convex-hull
        self.maxNumVerticesPerCH = 32
        # Minimum volume to add vertices to convex-hulls
        self.minVolumePerCH = 0.0001


    def run(self, in_path, out_path):

        cmd_line = '"{}" --input "{}" --resolution {} --depth {} --concavity {:g} --planeDownsampling {} --convexhullDownsampling {} --alpha {:g} --beta {:g} --gamma {:g} --pca {:b} --mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} --output "{}" --log "{}"'.format(
            self.vhacd_path,
            in_path,
            self.resolution,
            self.depth,
            self.concavity,
            self.planeDownsampling,
            self.convexhullDownsampling,
            self.alpha,
            self.beta,
            self.gamma,
            self.pca,
            self.mode == 'TETRAHEDRON',
            self.maxNumVerticesPerCH,
            self.minVolumePerCH,
            out_path,
            VHACD_LOG_FILE)

        vhacd_process = Popen(cmd_line, bufsize=-1, close_fds=True, shell=True)
        vhacd_process.wait()


class Graspable(object):
    def __init__(self, shapenet_path, output_path, categories_fn, nobjs=None):

        self.shapenet_path = shapenet_path
        self.output_path = output_path
        self.categories_fn = categories_fn
        self.nobjs = nobjs

        self.metadata = pd.read_csv(os.path.join(self.shapenet_path, 'metadata.csv'))

        with open(categories_fn, 'r') as f:
            categories = f.readlines()
            categories = [subcat.replace('#','').replace(',','').replace('\n','') for cat in categories for subcat in cat.split(' ') ]
            categories = [cat for cat in categories if cat is not '']
            categories = set(categories)

        graspable = self.metadata.loc[ [ len(set(str(x).split(',')).intersection(categories)) > 0 for x in self.metadata['category']] ]
        self.ids = [x.replace('wss.', '') for x in graspable['fullId']]
        shuffle(self.ids)
        self.ids = self.ids[:self.nobjs]

    def copy_objs(self, to_path):
        if not os.path.exists(to_path):
            os.makedirs(to_path)

        for idx in self.ids:
            obj_fn = os.path.join(self.shapenet_path, idx + '.obj')
            os.system('cp {} {}'.format(obj_fn, to_path))

    def compute_vhacds(self, to_path):
        if not os.path.exists(to_path):
            os.makedirs(to_path)

        vhacd = VHACD()

        for idx in self.ids:
            obj_fn = os.path.join(self.shapenet_path, idx + '.obj')
            vhacd_fn = os.path.join(to_path, idx + '_vhacd.obj')
            vhacd.run(obj_fn, vhacd_fn)

    def convert_to_vhacd(self, from_path, to_path):
        objs_fn = glob.glob(os.path.join(from_path, '*.obj'))
        vhacd = VHACD()
        for obj_fn in objs_fn:
            idx = obj_fn.replace('.obj','')
            vhacd_fn = os.path.join(to_path, idx + '_vhacd.obj')
            vhacd.run(obj_fn, vhacd_fn)
        pass

class Wavefront(object):

    def __init__(self, fn):
        self.vertices = np.array([], (None, 3))
        with open(fn, 'r') as f:
            for line in f.readlines():
                fields = line.split(' ')
                if fields[0] == 'v':
                    self.vertices = np.append(self.vertices, [[float(x) for x in fields[1:4]]], axis=0)

    @property
    def aabb(self):
        lower = np.min(self.vertices, axis=0)
        upper = np.max(self.vertices, axis=0)
        return np.vstack((lower, upper))

    @property
    def center(self):
        return np.mean(self.vertices, axis=0)

    @property
    def size(self):
        bb = self.aabb
        return np.abs(bb[1] - bb[0])

@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target

