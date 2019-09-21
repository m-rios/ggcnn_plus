import argparse
from utils import dataset as ds

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='path to hdf5')
parser.add_argument('factor', type=float, help='resampling factor')

args = parser.parse_args()

ds.subsample(args.dataset, args.factor)
