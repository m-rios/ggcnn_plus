import os
import glob
import pandas as pd
import argparse
from random import shuffle

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shapenet', default=os.environ['SHAPENET_PATH'])
    parser.add_argument('--output', default=os.environ['MODELS_PATH'])
    parser.add_argument('--nobjs', default=None, type=int)

    args = parser.parse_args()

    shpn = pd.read_csv(os.path.join(args.shapenet, 'metadata.csv'))

    with open('categories.txt','r') as f:
        categories = f.readlines()
        categories = [subcat.replace('#','').replace(',','').replace('\n','') for cat in categories for subcat in cat.split(' ') ]
        categories = [cat for cat in categories if cat is not '']

    categories = set(categories)
    graspable = shpn.loc[ [ len(set(str(x).split(',')).intersection(categories)) > 0 for x in shpn['category']] ]
    graspable_ids = [x.replace('wss.', '') for x in graspable['fullId']]
    shuffle(graspable_ids)
    graspable_ids = graspable_ids[:args.nobjs]


    if not os.path.exists(args.output):
        os.makedirs(args.output)
    os.chdir('/home/mario/Developer/v-hacd/build/linux/test')
    for graspable_id in graspable_ids:
        input_fn = os.path.join(args.shapenet, graspable_id + '.obj')
        output_fn = os.path.join(args.output, graspable_id + '_vhacd.obj')
        #vhacd.run(input_fn, output_fn)
        os.system('./testVHACD --input {} --output {}'.format(input_fn, output_fn))
