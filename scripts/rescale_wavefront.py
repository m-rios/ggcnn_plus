import argparse
import os
import glob
from simulator.utils import Wavefront

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to file or folder')
    parser.add_argument('factor', type=int, help='Rescaling factor')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        fns = glob.glob(os.path.join(args.path, '*.obj'))
        output_path = os.path.join(args.path, 'rescaled_{}'.format(args.factor))
    elif os.path.isfile(args.path):
        fns = [args.path]
        output_path = os.path.join(os.path.dirname(args.path), 'rescaled_{}'.format(args.factor))
    else:
        print 'Path is not a valid file or directory'
        exit(-1)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fn in fns:
        name = fn.split('/')[-1]
        obj = Wavefront(fn)
        obj.rescale(args.factor)
        obj.save(os.path.join(output_path, name))
