import argparse
import os
from utils import Graspable

parser = argparse.ArgumentParser()
parser.add_argument('--shapenet', default=os.environ['SHAPENET_PATH'], help='Path to shapenet')
parser.add_argument('--models', default=os.environ['MODELS_PATH'], help='path where selected models will be copied to')
parser.add_argument('--categories', default='/home/mario/Developer/msc-thesis/simulator/categories.txt', help='File listing categories to consider')
parser.add_argument('--nobjs', type=int, default=None, help='Number of objects to subsample')
parser.add_argument('--vhacd', action='store_true', help='create vhacd instead of copy')

args = parser.parse_args()

graspable = Graspable(args.shapenet, args.models, args.categories, args.nobjs)
if args.vhacd:
    graspable.convert_to_vhacd(args.models, args.models)
else:
    graspable.copy_objs(args.models)
