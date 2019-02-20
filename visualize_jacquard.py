#!/usr/bin/env python
import cv2
import os
import glob
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from skimage import io
plt.ion()

DATASET_PATH = '/Volumes/Data/Jacquard'

if __name__ == '__main__':
    root, dir_names, file_names = next(os.walk(DATASET_PATH))
    fig, ax = plt.subplots(1,3)
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
            rgb_img_2 = io.imread(os.path.join(obj_path, rgb_img_fn))
            stereo_depth_img = misc.imread(os.path.join(obj_path, stereo_depth_img_fn))
            stereo_depth_img_2 = io.imread(os.path.join(obj_path, stereo_depth_img_fn))
            perfect_depth_img = misc.imread(os.path.join(obj_path, perfect_depth_img_fn))

            #inpaint stereo

            stereo_depth_img_2 = cv2.copyMakeBorder(stereo_depth_img_2, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            mask = (stereo_depth_img_2 == -1).astype(np.uint8)

            # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
            scale = np.abs(stereo_depth_img_2).max()
            stereo_depth_img_2 = stereo_depth_img_2.astype(np.float32) / scale  # Has to be float32, 64 not supported.
            stereo_depth_img_2 = cv2.inpaint(stereo_depth_img_2, mask, 1, cv2.INPAINT_NS)

            # Back to original size and value range.
            stereo_depth_img_2 = stereo_depth_img_2[1:-1, 1:-1]
            stereo_depth_img_2 = stereo_depth_img_2 * scale

            fig.suptitle(obj_id+'_'+obj_dir)
            ax[0].imshow(rgb_img_2)
            ax[0].set_title('rgb')
            ax[1].imshow(perfect_depth_img)
            ax[1].set_title('perfect depth')
            ax[2].imshow(stereo_depth_img_2)
            ax[2].set_title('stereo depth')

            plt.show()
            while not plt.waitforbuttonpress():
                pass
        # depth_imgs_fn = glob.glob(os.path.join(obj_path, '*.tiff'))
        # rgb_imgs_fn = glob.glob(os.path.join(obj_path, '*.tiff'))

