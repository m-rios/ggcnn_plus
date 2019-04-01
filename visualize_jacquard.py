#!/usr/bin/env python
import argparse
import h5py
import pylab as plt
from ggcnn.dataset_processing.grasp import BoundingBoxes, BoundingBox
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()

if __name__ == '__main__':
    ds = h5py.File(args.dataset, 'r')
    depths = ds['test']['depth_inpainted'][:]
    bbs = ds['test']['bounding_boxes'][:]

    for idx in range(depths.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        gs = BoundingBoxes.load_from_array(bbs[idx])
        plt.imshow(depths[idx])
        for g in gs:
            g.as_grasp.plot(ax)
        plt.show()

