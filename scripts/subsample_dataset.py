import argparse
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='path to hdf5 dataset file')
parser.add_argument('--factor', default=0.01,type=float, help='percentage of the dataset to subsample')
parser.add_argument('--train', action='store_true', help='subsample the train dataset (defaults to test)')

args = parser.parse_args()

ds = h5py.File(args.dataset, mode='r')
subset_name = 'train' if args.train else 'test'

length = ds[subset_name]['img_id'].size
subsample_length = int(length * args.factor)

result_ds = h5py.File(args.dataset.replace('.hdf5', '') + '_subsample_{}.hdf'.format(args.factor), 'w')

selection_idx = np.random.choice(range(length), subsample_length)
pass
for field_name, field_value in zip(ds[subset_name].keys(), ds[subset_name].values()):
    result_ds[subset_name + '/' + field_name] = field_value[:subsample_length]
