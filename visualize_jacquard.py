#!/usr/bin/env python
import cv2
import os
import glob
import matplotlib.pyplot as plt
from scipy import misc 
plt.ion()

DATASET_PATH = 'data/datasets/jacquard_samples'

if __name__ == '__main__':
    root, dir_names, file_names = next(os.walk(DATASET_PATH))
    for obj_dir in dir_names:
        obj_path = os.path.join(root, obj_dir)

        # Isolate object instances
        objs = set()
        for fn in os.listdir(obj_path):
            objs.add(fn.split('_')[0])
        for obj_id in objs:
            rgb_img_fn = '{}_{}_RGB.png'.format(obj_id, obj_dir)
            stereo_depth_img_fn = '{}_{}_stereo_depth.tiff'.format(obj_id, obj_dir)
            perfect_depth_img_fn = '{}_{}_perfect_depth.tiff'.format(obj_id, obj_dir)


            rgb_img = misc.imread(os.path.join(obj_path, rgb_img_fn))
            stereo_depth_img = misc.imread(os.path.join(obj_path, stereo_depth_img_fn))
            perfect_depth_img = misc.imread(os.path.join(obj_path, perfect_depth_img_fn))

            # Normalize depth images to [0:255]
            #stereo_depth_img = (stereo_depth_img - stereo_depth_img.min())/(stereo_depth_img.max() - stereo_depth_img.min())

            plt.imshow(rgb_img)
            plt.show()
            while not plt.waitforbuttonpress():
                pass

            plt.imshow(perfect_depth_img)
            plt.show()
            while not plt.waitforbuttonpress():
                pass

            plt.imshow(stereo_depth_img)
            plt.show()
            while not plt.waitforbuttonpress():
                pass

        # depth_imgs_fn = glob.glob(os.path.join(obj_path, '*.tiff'))
        # rgb_imgs_fn = glob.glob(os.path.join(obj_path, '*.tiff'))

