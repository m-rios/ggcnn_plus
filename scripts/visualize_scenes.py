#!/usr/bin/env python
import argparse
import h5py
import pylab as plt
parser = argparse.ArgumentParser()
parser.add_argument('scene')
args = parser.parse_args()

if __name__ == '__main__':
    ds = h5py.File(args.scene, 'r')
    depths = ds['depth']
    # bbs = ds['test']['bounding_boxes']

    for idx in range(depths.shape[0]):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(depths[idx])
        plt.show()

