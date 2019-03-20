from dataset import Jacquard
from ggcnn.dataset_processing import grasp
from ggcnn.dataset_processing.image import Image, DepthImage

import datetime
import json
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mpldatacursor

PAD_TO = 1154 # Number of grasps

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=os.environ['JACQUARD_PATH'], help='Path to dataset')
parser.add_argument('--output', default=os.environ['PREPROCESSED_PATH'], help='Path to output dataset')
parser.add_argument('--perfect', action='store_true', help='Use perfect depth images [default False]')
parser.add_argument('--rotations', default=1, type=int, help='Number of random rotations for augmentation')
parser.add_argument('--zoom', action='store_true', help='Use zoom augmentation')
parser.add_argument('--visualize', action='store_true', help='Visualize only')
parser.add_argument('--fraction', default=1, type=float, help='Percent of data to consider')
parser.add_argument('--split', default=0.8, type=float, help='Train/test split')
parser.add_argument('--size', default=300, type=int, help='Size of input layer (always squared)')
args = parser.parse_args()


aug_factor = args.rotations
dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
output_dataset_fn = os.path.join(args.output, '{}.hdf5'.format(dt))
output_dataset_description_fn = os.path.join(args.output, '{}_desc.json'.format(dt))
depth_type = ['stereo_depth', 'perfect_depth'][args.perfect]
subsets = ['train', 'test']

jaq = Jacquard(args.dataset, fraction=args.fraction, split=args.split)

if not args.visualize:
    description = {}
    for arg in vars(args):
        description[arg] = getattr(args, arg)
    description['creation_date'] = dt
    with open(output_dataset_description_fn,'w') as json_description:
        json.dump(description, json_description, indent=2)

    output_ds = h5py.File(output_dataset_fn, 'w')

    fields = [
        'img_id',
        'depth_inpainted',
        'bounding_boxes',
        'grasp_points_img',
        'angle_img',
        'grasp_width'
    ]

    sizes = {
            'img_id':           (),
            'rgb':              (args.size, args.size, 3),
            'depth_inpainted':  (args.size, args.size),
            'bounding_boxes':   (PAD_TO, 4, 2),
            'grasp_points_img': (args.size, args.size),
            'angle_img':        (args.size, args.size),
            'grasp_width':      (args.size, args.size)
    }

    types = {
        'angle_img': 'float64',
        'bounding_boxes': 'int64',
        'depth_inpainted': 'float32',
        'grasp_points_img': 'float64',
        'grasp_width': 'float64',
        'img_id': '|S32',
        'rgb': 'uint8',
    }

    for subset in subsets:
        for field in fields:
            subset_sz = (getattr(jaq, subset+'_keys').size * aug_factor,) + sizes[field]
            output_ds.create_dataset(subset + '/' + field, subset_sz,
                    dtype=types[field])


for subset in subsets:
    idx = 0
    for key in getattr(jaq, subset+'_keys'):
        print('Processing {}'.format(key))
        scene = jaq[key]

        depth_img_base = DepthImage(scene[depth_type])
        depth_img_base.inpaint(missing_value=-1)
        bounding_boxes_base = grasp.BoundingBoxes(scene['bounding_boxes'])
        center = bounding_boxes_base.center

        for i in range(args.rotations):
            if args.rotations > 1:
                angle = np.random.random() * 2 * np.pi - np.pi
            else:
                angle = 0

            depth = depth_img_base.rotated(angle, center)
            bbs = bounding_boxes_base.copy()
            bbs.rotate(angle, center)

            left = max(0, min(center[1] - args.size // 2, depth.shape[1] - args.size))
            right = min(depth.shape[1], left + args.size)

            top = max(0, min(center[0] - args.size // 2, depth.shape[0] - args.size))
            bottom = min(depth.shape[0], top + args.size)

            depth.crop((top, left), (bottom, right))
            bbs.offset((-top, -left))

            if args.zoom:
                zoom_factor = np.random.uniform(0.4, 1.0)
                depth.zoom(zoom_factor)
                depth.zoom(zoom_factor)
                bbs.zoom(zoom_factor, (args.size//2, args.size//2))

            pos_img, ang_img, width_img = bbs.draw(depth.shape)

            if args.visualize:
                rgb = Image(scene['rgb'])
                rgb.crop((top, left), (bottom, right))
                fig, ax = plt.subplots(1,2)
                mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'),formatter='i, j = {i}, {j}\nz ={z:.02g}'.format)
                ax[0].clear() # remove old bb
                fig.suptitle(key)
                rgb.show(ax[0])
                bbs.show(ax[0])
                ax[0].set_xlim((0, args.size))
                ax[0].set_ylim((args.size), 0)
                ax[0].set_title('rgb')
                mp = depth.show(ax[1])
                ax[1].set_title('depth')
                plt.show()
            else:
                output_ds[subset]['img_id'][idx] = scene['img_id']
                output_ds[subset]['depth_inpainted'][idx] = depth.img
                output_ds[subset]['bounding_boxes'][idx] = bbs.to_array(pad_to=PAD_TO)
                output_ds[subset]['grasp_points_img'][idx] = pos_img
                output_ds[subset]['angle_img'][idx] = ang_img
                output_ds[subset]['grasp_width'][idx] = width_img
                idx += 1


