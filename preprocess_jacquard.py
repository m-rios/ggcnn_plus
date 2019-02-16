#!/usr/bin/env python

from getpass import getpass
from ggcnn.dataset_processing import grasp
from ggcnn.dataset_processing.image import Image, DepthImage
from scipy import misc
from skimage import io
from skimage.transform import resize
import argparse
import copy
import datetime
import glob
import h5py
import json
import matplotlib.pyplot as plt
import mpldatacursor
import numpy as np
import os
import sys

DATASET_PATH = 'data/datasets/raw/jacquard_samples'  # Path to the original dataset files
OUTPUT_PATH = 'data/datasets/preprocessed'  # Destination of the pre-processed dataset
TRAIN_SPLIT = 0.0
TEST_IMAGES = None # List of obj_ids to use for testing
RANDOM_ROTATIONS = 1
RANDOM_ZOOM = False
OUTPUT_IMG_SIZE = (300, 300)
#OUTPUT_IMG_SIZE = (1024, 1024)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', help='visualize only')
    parser.add_argument('-d', default=DATASET_PATH, help='path to dataset')
    parser.add_argument('-o', default=OUTPUT_PATH, help='ouput path of processed dataset')
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
    if TEST_IMAGES:
        description['test_images'] = TEST_IMAGES
    else:
        description['train_split'] = TRAIN_SPLIT
    description['stereo'] = USE_STEREO
    description['img_size'] = OUTPUT_IMG_SIZE
    description['augmentation'] = {'random_rotations': RANDOM_ROTATIONS,
            'random_zoom': RANDOM_ZOOM}
    description['creation_date'] = dt

    if VISUALIZE_ONLY:
        plt.ion()
        fig, ax = plt.subplots(1,2)
        mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),formatter='i, j = {i}, {j}\nz ={z:.02g}'.format)

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
            #hist_base = histogram(depth_img_base)
            depth_img_base.inpaint(missing_value=-1)
            #hist_inpainted = histogram(depth_img_base)
            bounding_boxes_base = grasp.BoundingBoxes.load_as_jacquard(os.path.join(obj_class_path,grasp_fn))
            center = bounding_boxes_base.center
            #plt.hist(hist_base)

            # Split train/test
            ds_output = 'train'
            if TEST_IMAGES:
                if int(obj_id) in TEST_IMAGES:
                    ds_output = 'test'
            elif np.random.rand() > TRAIN_SPLIT:
                ds_output = 'test'
            ds = dataset[ds_output]

            for i in range(RANDOM_ROTATIONS):
                # Skip rotation if no augmentation
                if RANDOM_ROTATIONS > 1:
                    angle = np.random.random() * 2 * np.pi - np.pi
                else:
                    angle = 0
                rgb = rgb_img_base.rotated(angle, center)
                depth = depth_img_base.rotated(angle, center)
                bbs = bounding_boxes_base.copy()
                bbs.rotate(angle, center)

                left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, rgb.shape[1] - OUTPUT_IMG_SIZE[1]))
                right = min(rgb.shape[1], left + OUTPUT_IMG_SIZE[1])

                top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, rgb.shape[0] - OUTPUT_IMG_SIZE[0]))
                bottom = min(rgb.shape[0], top + OUTPUT_IMG_SIZE[0])

                rgb.crop((top, left), (bottom, right))
                depth.crop((top, left), (bottom, right))
                bbs.offset((-top, -left))

                if RANDOM_ZOOM:
                    zoom_factor = np.random.uniform(0.4, 1.0)
                    rgb.zoom(zoom_factor)
                    depth.zoom(zoom_factor)
                    bbs.zoom(zoom_factor, (OUTPUT_IMG_SIZE[0]//2, OUTPUT_IMG_SIZE[1]//2))

                depth.normalise()
                pos_img, ang_img, width_img = bbs.draw(depth.shape)


                if VISUALIZE_ONLY:
                    ax[0].clear() # remove old bb
                    fig.suptitle(obj_id+'_'+obj_class)
                    rgb.show(ax[0])
                    bbs.show(ax[0])
                    ax[0].set_title('rgb')
                    depth.show(ax[1])
                    ax[1].set_title('depth')
                    #plt.savefig(str(i)+'_'+obj_id+'_'+obj_class+'.png', format='png')
                    plt.show()
                    while not plt.waitforbuttonpress():
                        pass
                else:
                    ds['img_id'].append(int(obj_id))
                    ds['rgb'].append(rgb.img)
                    ds['depth_inpainted'].append(depth.img)
                    ds['bounding_boxes'].append(bbs.to_array(pad_to=200))
                    ds['grasp_points_img'].append(pos_img)
                    ds['angle_img'].append(ang_img)
                    ds['grasp_width'].append(width_img)


    if not VISUALIZE_ONLY:
        with open(output_dataset_description_fn,'w') as json_description:
            json.dump(description, json_description, indent=2)
        with h5py.File(output_dataset_fn,'w') as f:
            for tt_name in dataset:
                for ds_name in dataset[tt_name]:
                    import ipdb; ipdb.set_trace() # BREAKPOINT
                    f.create_dataset('{}/{}'.format(tt_name, ds_name),
                            data=np.array(dataset[tt_name][ds_name]))
