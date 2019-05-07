import h5py
import argparse
import glob
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes')

    args = parser.parse_args()

    scene_fns = glob.glob(args.scenes + '/*.csv')

    scenes_ds = h5py.File(args.scenes+'/scenes.hdf5', 'w')
    tp = h5py.special_dtype(vlen=bytes)
    scenes_ds.create_dataset('name', (len(scene_fns),), dtype=tp)
    scenes_ds.create_dataset('rgb', (len(scene_fns),300, 300, 3), dtype='u8')
    scenes_ds.create_dataset('depth', (len(scene_fns),300, 300), dtype=np.float64)
    scenes_ds.create_dataset('scene', (len(scene_fns),), dtype=tp)

    for idx, scene_fn in enumerate(scene_fns):
        scene_name = scene_fn.split('/')[-1].replace('_scene.csv', '')
        np_fn = scene_fn.replace('_scene.csv','_300_300.npy')
        imgs = np.load(np_fn)
        scenes_ds['name'][idx] = scene_name
        scenes_ds['rgb'][idx] = imgs[:,:,:3]
        scenes_ds['depth'][idx] = imgs[:,:,3]
        with open(scene_fn,'r') as f:
            scenes_ds['scene'][idx] = f.read()



