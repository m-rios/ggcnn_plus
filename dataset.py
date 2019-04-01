"""
Class to easily handle Jacquard dataset
"""
from skimage import io
from ggcnn.dataset_processing.grasp import BoundingBox, BoundingBoxes, Grasp

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Jacquard:

    def __init__(self, path, fraction=1, split=0.8):
        self.path = path
        self.split = split
        self.keys = ['_'.join(x.split('/')[-1].split('_')[0:2]) for x in glob.iglob(os.path.join(path, '*/*_grasps.txt'))]
        self.keys = np.random.choice(np.array(self.keys), int(len(self.keys)*fraction))
        self._train_idx, self._test_idx = np.split(np.random.permutation(np.arange(self.size)), [int(self.size*split)])

    @property
    def size(self):
        return self.keys.size

    @property
    def train_keys(self):
        return self.keys[self._train_idx]

    @property
    def test_keys(self):
        return self.keys[self._test_idx]

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
        for key in self.keys:
            yield self[key]

    def _load_grasps(self, key):
        scene, cls = key.split('_')
        fname = os.path.join(self.path, cls, '{}_grasps.txt'.format(key))
        bbs = []
        grasps = pd.read_csv(fname, delimiter=';', names='x;y;theta;opening;size'.split(';'))
        grasps.drop_duplicates(subset=['x','y'], inplace=True)
        for id, grasp in grasps.iterrows():
            x, y, theta, width, size = grasp
            angle = -np.radians(float(theta)) # Angle is horizontally mirrored
            gs = Grasp((y, x), angle, float(width), float(size))
            bbs.append(gs.as_bb)

        return bbs

    def plot(self, key):
        data = self[key]
        depth = data['stereo_depth']
        rgb = data['rgb']
        bbs = BoundingBoxes(data['bounding_boxes'])

        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('RGB')
        ax.imshow(rgb)

        for bb in bbs:
            g = bb.as_grasp
            g.plot(ax)

        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Depth')
        ax.imshow(depth)

        for bb in bbs:
            g = bb.as_grasp
            g.plot(ax)

        plt.show()

if __name__ == '__main__':
    d = Jacquard('data/datasets/raw/jacquard_samples')
    #for j in d:
    #    print j['img_id']
    #    print len(j['bounding_boxes'])
    #print 'train'
    #print d.train_keys
    #print 'test'
    #print d.test_keys

    for k in d.keys:
        d.plot(k)
