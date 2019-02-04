#!/usr/bin/env python

from getpass import getpass
from scipy import misc
import argparse
import datetime
import glob
import os
import sys
import copy

DATASET_PATH = 'data/datasets/raw/jacquard_samples'  # Path to the original dataset files
OUTPUT_PATH = 'data/datasets/preprocessed'  # Destination of the pre-processed dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='visualize only')

    args = parser.parse_args()
    VISUALIZE_ONLY = args.v

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
            stereo_depth_img_fn = '{}_{}_stereo_depth.tiff'.format(obj_id, obj_class)
            perfect_depth_img_fn = '{}_{}_perfect_depth.tiff'.format(obj_id, obj_class)


