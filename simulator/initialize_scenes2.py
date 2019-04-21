import glob
import argparse
import os
import numpy as np
from simulator import Simulator
import simulator as s

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=os.environ['MODELS_PATH'], help='path to folder with obj files')
    parser.add_argument('--scenes', default=os.environ['GGCNN_SCENES_PATH'], help='path to folder scenes will be saved')
    parser.add_argument('--width', default=300, type=int, help='width of image')
    parser.add_argument('--height', default=300, type=int, help='height of image')
    parser.add_argument('--nscenes', default=5, type=int, help='Number of scenes per object')
    parser.add_argument('--gui', action='store_true')

    args = parser.parse_args()

    obj_fns = set(glob.glob(os.path.join(args.models, '*.obj'))) - set(glob.glob(os.path.join(args.models, '*_vhacd.obj')))

    if not os.path.exists(args.scenes):
        os.makedirs(args.scenes)

    assert(len(obj_fns) > 0)
    timestep = 1./240.
    sim = Simulator(gui=args.gui, use_egl=False, epochs=int(5/timestep), timestep=timestep,
            stop_th=1e-6, debug=args.gui)
    sim.cam.width = args.width
    sim.cam.height = args.height

    for obj_fn in obj_fns:
        for scene_n in range(args.nscenes):
            print('Processing: {} {} '.format(scene_n, obj_fn))
            scale = sim.read_scale(obj_fn.split('/')[-1].replace('.obj',''))
            sim.load(obj_fn, ori=np.random.rand(3) * 2*np.pi, scale=scale)
            sim.run(autostop=True)
            if sim.is_stable():
                obj_name = obj_fn.split('/')[-1].split('.')[-2]
                world_fn = str(scene_n) + '_' + obj_name + '_scene.csv'
                sim.save(os.path.join(args.scenes, world_fn))
                sim.cam.target = sim.get_clutter_center().tolist()
                rgb, depth = sim.cam.snap()
                input = np.zeros((args.height, args.width, 4))
                input[:,:,0:3] = rgb[:,:,0:3]
                input[:,:,3] = depth
                np.save('{}/{}_{}_{}_{}.npy'.format(args.scenes,scene_n, obj_name, args.width, args.height), input)
            sim.reset()




