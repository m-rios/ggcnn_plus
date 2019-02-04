#!/usr/bin/env python

from getpass import getpass
from ggcnn.dataset_processing import grasp
from ggcnn.dataset_processing.image import Image, DepthImage
from scipy import misc
from skimage import io
import argparse
import copy
import datetime
import glob
import matplotlib.pyplot as plt
import os
import sys

DATASET_PATH = 'data/datasets/raw/jacquard_samples'  # Path to the original dataset files
OUTPUT_PATH = 'data/datasets/preprocessed'  # Destination of the pre-processed dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='visualize only')
    parser.add_argument('-d', default=DATASET_PATH, help='path to dataset')
    parser.add_argument('-o', default=DATASET_PATH, help='ouput path of processed dataset')
    parser.add_argument('--perfect', action='store_true', help='use perfect depth images [default False]')

    args = parser.parse_args()
    VISUALIZE_ONLY = args.v
    DATASET_PATH = args.d
    OUTPUT_PATH = args.o
    USE_STEREO = not args.perfect

    # Open dataset path
    try:
        dataset_root, obj_classes, dataset_fns = next(os.walk(DATASET_PATH))
        obj_classes.sort()
    except:
        print('Could not find path: {}'.format(DATASET_PATH))
        sys.exit(0)

    # Create output path if it doesnt exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    fields = [
        'img_id',
        'rgb',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width'
    ]

    # Empty datatset.
    dataset = {
        'test':  dict([(f, []) for f in fields]),
        'train': dict([(f, []) for f in fields])
    }

    # Output file names
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    output_dataset_fn = os.path.join(OUTPUT_PATH, '{}.hdf5'.format(dt))
    output_dataset_description_fn = os.path.join(OUTPUT_PATH, '{}_desc.json'.format(dt))

    description = {}
    description['original_dataset'] = DATASET_PATH.split('/')[-1]
    description['stereo'] = USE_STEREO

    if VISUALIZE_ONLY:
        plt.ion()
        fig, ax = plt.subplots(1,2)

    for obj_class in obj_classes:
        obj_class_path = os.path.join(dataset_root, obj_class)

        # Isolate object instances
        obj_ids = set()
        for obj_fn in os.listdir(obj_class_path):
            obj_ids.add(obj_fn.split('_')[0])

        # Preprocess object
        obj_ids = list(obj_ids)
        obj_ids.sort()
        for obj_id in obj_ids:
            print('Processing {}_{}'.format(obj_id, obj_class))

            rgb_img_fn = '{}_{}_RGB.png'.format(obj_id, obj_class)
            if USE_STEREO:
                depth_img_fn = '{}_{}_stereo_depth.tiff'.format(obj_id, obj_class)
            else:
                depth_img_fn = '{}_{}_perfect_depth.tiff'.format(obj_id, obj_class)
            grasp_fn = '{}_{}_grasps.txt'.format(obj_id, obj_class)

            rgb_img_base = Image(io.imread(os.path.join(obj_class_path, rgb_img_fn)))
            depth_img_base = DepthImage(io.imread(os.path.join(obj_class_path, depth_img_fn)))
            depth_img_base.inpaint(missing_value=-1)
            bounding_boxes_base = grasp.BoundingBoxes.load_as_jacquard(os.path.join(obj_class_path,grasp_fn))

            if VISUALIZE_ONLY:
                ax[0].clear() # remove old bb
                fig.suptitle(obj_id+'_'+obj_class)
                rgb_img_base.show(ax[0])
                bounding_boxes_base.show(ax[0])
                ax[0].set_title('rgb')
                depth_img_base.show(ax[1])
                ax[1].set_title('depth')
                plt.show()
                while not plt.waitforbuttonpress():
                    pass
                continue
