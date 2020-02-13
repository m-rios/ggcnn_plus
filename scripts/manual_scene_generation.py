import glob
import datetime
import argparse
import os
import numpy as np
from simulator.simulator import Simulator
import h5py

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=os.environ['MODELS_PATH'], help='path to folder with obj files')
    parser.add_argument('--scenes', default=os.environ['GGCNN_SCENES_PATH'], help='path to folder scenes will be saved')
    parser.add_argument('--name', default='manually_generated_scenes', type=str, help='Name of the hdf5 file')
    parser.add_argument('--width', default=300, type=int, help='width of image')
    parser.add_argument('--height', default=300, type=int, help='height of image')
    parser.add_argument('--nscenes', default=5, type=int, help='Number of scenes per object')
    parser.add_argument('--subsample', default=None, type=float)
    parser.add_argument('--default-scale', action='store_true')

    args = parser.parse_args()

    obj_fns = set(glob.glob(os.path.join(args.models, '*.obj'))) - set(
        glob.glob(os.path.join(args.models, '*_vhacd.obj')))
    assert (len(obj_fns) > 0)

    if not os.path.exists(args.scenes):
        os.makedirs(args.scenes)

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    scenes_ds = os.path.join(args.scenes, dt + '_' + args.name + '.hdf5')
    scenes_ds = h5py.File(scenes_ds, 'w')
    tp = h5py.special_dtype(vlen=bytes)
    scenes_ds.create_dataset('name', (len(obj_fns) * args.nscenes,), dtype=tp)
    scenes_ds.create_dataset('rgb', (len(obj_fns) * args.nscenes, args.height, args.width, 3), dtype='u8')
    scenes_ds.create_dataset('depth', (len(obj_fns) * args.nscenes, args.height, args.width), dtype=np.float64)
    scenes_ds.create_dataset('scene', (len(obj_fns) * args.nscenes,), dtype=tp)

    sim = Simulator(gui=True, use_egl=False, debug=True)
    sim.cam.width = args.width
    sim.cam.height = args.height

    for var in vars(sim.cam).items():
        scenes_ds.attrs[var[0]] = var[1]

    for i, obj_fn in enumerate(obj_fns):
        scene_it = iter(range(args.nscenes))
        scene_n = next(scene_it, None)
        while scene_n is not None:
            print('Processing: {} {} '.format(scene_n, obj_fn))
            idx = i * args.nscenes + scene_n
            if not args.default_scale:
                scale = sim.read_scale(obj_fn.split('/')[-1].replace('.obj', ''), args.models)
            else:
                scale = 1
            sim.load(obj_fn, scale=scale)
            stop = sim.run_generate_scene()

            obj_name = obj_fn.split('/')[-1].split('.')[-2]
            scenes_ds['name'][idx] = str(scene_n) + '_' + obj_name
            scenes_ds['scene'][idx] = sim.get_state()

            sim.cam.target = sim.get_clutter_center().tolist()
            rgb, depth = sim.cam.snap()
            scenes_ds['rgb'][idx] = rgb[:, :, :3]
            scenes_ds['depth'][idx] = depth

            scene_n = next(scene_it, None)

            if stop:
                scenes_ds.flush()
                exit(0)

            sim.reset()
