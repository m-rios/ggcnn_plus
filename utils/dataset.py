"""
Class to easily handle Jacquard dataset
"""
from skimage import io
from ggcnn.dataset_processing.grasp import BoundingBoxes, Grasp
from keras.utils import Sequence

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py


def subsample(dataset_fn, ratio):
    """
        subsample a preprocessed dataset (.hdf5).
    """
    from core import network as net
    output_fn = dataset_fn.replace('.hdf5', '_resampled_{}.hdf5'.format(ratio))
    input_ds = h5py.File(dataset_fn, 'r')
    output_ds = h5py.File(output_fn, 'w')
    for subset in input_ds.iterkeys():
        for field in input_ds[subset].iterkeys():
            path = subset + '/' + field
            output_ds.create_dataset(path, data=input_ds[path])

    for subset in output_ds.iterkeys():
        path = subset + '/depth_inpainted'
        for img_idx in range(output_ds[path].shape[0]):
            output_ds[path][img_idx] = net.subsample(output_ds[path][img_idx], ratio)


class DatasetGenerator(Sequence):
    def __init__(self, path, batch_sz, dataset='train'):
        self.ds = h5py.File(path, 'r')[dataset]
        self.n_samples = self.ds['img_id'].size
        self.batch_sz = batch_sz

    def __len__(self):
        return np.ceil(self.n_samples/float(self.batch_sz))

    def __getitem__(self, idx):

        fr = idx*self.batch_sz
        to = (idx+1)*self.batch_sz

        depth = np.expand_dims(np.array(self.ds['depth_inpainted'][fr:to]), -1)
        point = np.expand_dims(np.array(self.ds['grasp_points_img'][fr:to]), -1)
        angle = np.array(self.ds['angle_img'][fr:to])
        cos = np.expand_dims(np.cos(2*angle), -1)
        sin = np.expand_dims(np.sin(2*angle), -1)
        grasp_width = np.expand_dims(np.array(self.ds['grasp_width'][fr:to]), -1)
        grasp_width = np.clip(grasp_width, 0, 150)/150.0

        batch_x = depth
        batch_y = [point, cos, sin, grasp_width]

        return batch_x, batch_y


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
            'mask': io.imread(base_fn + '_mask.png').astype(np.bool),
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
        fig.suptitle(key)
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
