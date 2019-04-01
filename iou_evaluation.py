import argparse
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import network as net

from keras.models import load_model
from skimage.filters import gaussian
from ggcnn.dataset_processing.grasp import BoundingBoxes
from ggcnn.dataset_processing import grasp
from ggcnn.dataset_processing.image import DepthImage
from dataset import Jacquard

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to directory containing model epochs')
parser.add_argument('--dataset', help='path to dataset descrition file')
parser.add_argument('--results', default=os.environ['RESULTS_PATH'], help='path to directory where results should be saved')
parser.add_argument('--perfect', action='store_true', help='Use perfect depth images [default False]')
parser.add_argument('--n_grasps', default=1, type=int, help='Number of grasps to predict per image')
parser.add_argument('--miniou', default=0.25, type=float, help='Min iou to consider successful')
parser.add_argument('--saveviz', action='store_true', help='if true saves a visualization of network output')
args = parser.parse_args()

model_fns = glob.glob(os.path.join(args.model, '*.hdf5'))
assert len(model_fns) > 0, 'No model files were found'
model_name = model_fns[0].split('/')[-2]

if args.dataset:
    assert os.path.isfile(args.dataset) and args.dataset.split('.')[-1] == 'json'
    with open(args.dataset) as f:
        scenes = json.load(f)['test_ids']
else:
    raise NotImplementedError

jaq = Jacquard(os.environ['JACQUARD_PATH'])
input_sz = load_model(model_fns[0]).input.shape.as_list()[1:3]
depth_type = ['stereo_depth', 'perfect_depth'][args.perfect]
depths = np.zeros(([len(scenes)] + input_sz))
bbs = []

for idx, scene in enumerate(scenes):
    j_data = jaq[scene]
    depth = DepthImage(j_data[depth_type])
    depth.inpaint(missing_value=-1)
    depth.img -= depth.img.mean()
    gt = grasp.BoundingBoxes(j_data['bounding_boxes'])

    center = gt.center
    left = max(0, min(center[1] - input_sz[1] // 2, depth.shape[1] - input_sz[1]))
    right = min(depth.shape[1], left + input_sz[1])

    top = max(0, min(center[0] - input_sz[0] // 2, depth.shape[0] - input_sz[0]))
    bottom = min(depth.shape[0], top + input_sz[0])
    depth.crop((top, left), (bottom, right))
    depths[idx] = depth.img
    gt.offset((-top, -left))
    bbs.append(gt.to_array())

depth = np.expand_dims(depths, 3)
bbs = np.array(bbs)

save_path = os.path.join(args.results, model_name, 'iou')
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_fn = os.path.join(save_path, 'iou.txt')
results_f = open(results_fn, 'w')

for model_fn in model_fns:
    epoch = model_fn.split('_')[-2]
    epoch_path = os.path.join(save_path, epoch)
    if args.saveviz and not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    print('Evaluating epoch ' + epoch)
    network = net.Network(model_fn)
    positions, angles, widths = network.predict(depths, subtract_mean=False)

    succeeded, failed = net.calculate_iou_matches(positions, angles, bbs,
            args.n_grasps, widths, args.miniou)

    if args.saveviz:
        for idx in succeeded:
            net.save_output_plot(depth[idx], positions[idx], angles[idx],
                    widths[idx], epoch_path + '/s_{}.png'.format(idx),
                    ground_truth=bbs[idx])
        for idx in failed:
            net.save_output_plot(depth[idx], positions[idx], angles[idx],
                    widths[idx], epoch_path + '/f_{}.png'.format(idx),
                    ground_truth=bbs[idx])

    s = len(succeeded)
    f = len(failed)
    sr = float(s)/float(s+f)

    results_f.write('Epoch {}: {}%\n'.format(epoch, sr))

