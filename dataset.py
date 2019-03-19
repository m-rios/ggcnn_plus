"""
Class to easily handle Jacquard dataset
"""
from skimage import io
from ggcnn.dataset_processing.grasp import BoundingBox

import glob
import os
import pandas as pd
import numpy as np

class Jacquard:

    def __init__(self, path):
        self.path = path
        self.size = sum(1 for _ in glob.iglob(os.path.join(path, '*/*_grasps.txt')))
        self.fields = [
            'img_id',
            'rgb',
            'stereo_depth',
            'perfect_depth',
            'bounding_boxes',
        ]
        sizes = {
            'img_id': (self.size,),
            'rgb': (self.size, 1024, 1024, 3),
            'stereo_depth': (self.size, 1024, 1024),
            'perfect_depth': (self.size, 1024, 1024),
            'bounding_boxes': (self.size, 4, 2)
        }
        types = {
            'img_id': '|S32',
            'rgb': 'uint8',
            'stereo_depth': 'float32',
            'perfect_depth': 'float32',
            'bounding_boxes': 'int64'
        }

    def __getitem__(self, key):
        scene, cls = key.split('_')
        assert os.path.exists(os.path.join(self.path, cls, key + '_grasps.txt')), 'Key \'{}\' is not valid'.format(key)
        base_fn = os.path.join(self.path, cls, key)
        item = {
            'img_id': key,
            'rgb': io.imread(base_fn + '_RGB.png'),
            'stereo_depth': io.imread(base_fn + '_stereo_depth.tiff'),
            'perfect_depth': io.imread(base_fn + '_perfect_depth.tiff'),
            'bounding_boxes': self._load_grasps(key)
        }
        return item

    def __iter__(self):
        for grasp_fn in glob.iglob(os.path.join(self.path, '*/*_grasps.txt')):
            key = grasp_fn.split('/')[-1].split('_grasps.txt')[0]
            yield self[key]

    def _load_grasps(self, key):
        scene, cls = key.split('_')
        fname = os.path.join(self.path, cls, '{}_grasps.txt'.format(key))
        bbs = []
        grasps = pd.read_csv(fname, delimiter=';', names='x;y;theta;opening;size'.split(';'))
        grasps.drop_duplicates(subset=['x','y'], inplace=True)
        for id, grasp in grasps.iterrows():
            x, y, theta, width, size = grasp

            theta = np.radians(float(theta))
            # Vertices w.r.t. grasp center
            x_off = width/2.0
            y_off = size/2.0
            bb = np.array([
                [x_off, y_off ,1],
                [-x_off, y_off ,1],
                [-x_off, -y_off ,1],
                [x_off, -y_off ,1]
                ]).T
            assert(bb.shape == (3,4))
            # Transformation matrix to image frame of reference
            T = np.array([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [x, y, 1]
                ]).T
            # Apply transformation and swap coordinates for row/col
            # coordinates
            bb = np.matmul(T, bb).astype(int)
            bb = bb[[1,0],:].T

            assert(bb.shape == (4,2))

            bbs.append(BoundingBox(bb))

        return bbs

if __name__ == '__main__':
    d = Jacquard('data/datasets/raw/jacquard_samples')
    for j in d:
        print j['img_id']
        print len(j['bounding_boxes'])

