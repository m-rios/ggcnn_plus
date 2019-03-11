import glob
import argparse
import os
from simulator import Simulator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to folder with obj files')
    parser.add_argument('export', help='path to folder scenes will be saved')

    args = parser.parse_args()

    obj_fns = set(glob.glob(os.path.join(args.path, '*.obj'))) - set(glob.glob(os.path.join(args.path, '*_vhacd.obj')))

    if not os.path.exists(args.export):
        os.makedirs(args.export)

    assert(len(obj_fns) > 0)

    sim = Simulator(gui=True, stop_th=1e-6, debug=True)

    for obj_fn in obj_fns:
        sim.load(obj_fn)
        sim.run()
        if sim.is_stable():
            obj_name = obj_fn.split('/')[-1].split('.')[-2]
            world_fn = obj_name + '_scene.world'
            sim.save(os.path.join(args.export, world_fn))
        sim.reset()




