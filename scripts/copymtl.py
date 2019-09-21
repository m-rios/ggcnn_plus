import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('_from')
parser.add_argument('to')

args = parser.parse_args()

fns = glob.glob(os.path.join(args.to, '*.obj'))

ids = map(lambda x: x.split('/')[-1].replace('.obj','').replace('_vhacd', ''), fns)
ids = list(set(ids))
for id in ids:
    fn = id + '.mtl'
    shutil.copyfile(os.path.join(args._from, fn), os.path.join(args.to, fn))
