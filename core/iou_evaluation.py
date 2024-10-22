import argparse
import os
import glob
from core import network as net
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to directory containing model epochs')
parser.add_argument('dataset', help='path to hdf5 dataset file')
parser.add_argument('--results', default=os.environ['RESULTS_PATH'], help='path to directory where results should be saved')
parser.add_argument('--n_grasps', default=1, type=int, help='Number of grasps to predict per image')
parser.add_argument('--miniou', default=0.25, type=float, help='Min iou to consider successful')
parser.add_argument('--saveviz', action='store_true', help='if true saves a visualization of network output')
parser.add_argument('--epochs', nargs='+',default=None, type=int, help='epochs to evaluate')
parser.add_argument('--max_scenes', default=None, type=int, help='Max number of scenes to compare with (Default uses all available)')
args = parser.parse_args()

model_fns = glob.glob(os.path.join(args.model, '*.hdf5'))
assert len(model_fns) > 0, 'No model files were found'
model_name = model_fns[0].split('/')[-2]

ds = h5py.File(args.dataset, 'r')
scenes = ds['test']['img_id'][:args.max_scenes]
depth = ds['test']['depth_inpainted'][:args.max_scenes]
bbs = ds['test']['bounding_boxes'][:args.max_scenes]


save_path = os.path.join(args.results, model_name, 'iou')
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_fn = os.path.join(save_path, 'iou.txt')
results_f = open(results_fn, 'w')

results_f.write('Min IOU: {}\n'.format(args.miniou))
results_f.write('Test dataset size: {}\n'.format(scenes.size() if args.max_scenes is None else args.max_scenes))

for model_fn in model_fns:
    epoch = model_fn.split('_')[-2]
    if args.epochs is not None and int(epoch) not in args.epochs:
        continue
    epoch_path = os.path.join(save_path, epoch)
    if args.saveviz and not os.path.exists(epoch_path):
        os.makedirs(epoch_path)
    print('Evaluating epoch ' + epoch)
    network = net.Network(model_fn=model_fn)
    positions, angles, widths = network.predict(depth, subtract_mean=False)

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
    results_f.flush()

