import simulator as s
import numpy as np
import pylab as plt
import argparse
import os
import glob
from utils.dataset import Jacquard

roll = 0.
pitch = 0.
yaw = 0.
new_press = False

selected = False
discarded = False

def press(event):
    global selected, new_press, roll, pitch, yaw, discarded
    key = event.key
    if key == 'enter':
        selected = True
    if key == '.':
        discarded = True
    elif key == '7':
        roll += np.pi/2.
    elif key == '9':
        roll -= np.pi/2.
    elif key == '4':
        pitch += np.pi/2.
    elif key == '6':
        pitch -= np.pi/2.
    elif key == '1':
        yaw += np.pi/2.
    elif key == '3':
        yaw -= np.pi/2.
    new_press = True
    print [roll, pitch, yaw]

def depth_range(depth, mask):
    foreground = depth[mask]
    background = depth[np.logical_not(mask)]
    return np.abs(foreground.mean() - background.mean()).astype(np.float32)

def wait_keypress():
    global new_press
    while not new_press:
        plt.pause(0.01)
    new_press = False

def save(obj_name, scene_path, width, height):
    world_fn = obj_name + '_scene.csv'
    sim.save(os.path.join(scene_path, world_fn))
    sim.cam.target = sim.get_clutter_center().tolist()
    rgb, depth = sim.cam.snap()
    input = np.zeros((args.height, args.width, 4))
    input[:,:,0:3] = rgb[:,:,0:3]
    input[:,:,3] = depth
    np.save('{}/{}_{}_{}.npy'.format(scene_path, obj_name, width, height), input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',default=os.environ['MODELS_PATH'], help='path to models')
    parser.add_argument('--scenes',default=os.environ['GGCNN_SCENES_PATH'], help='path to output scenes')
    parser.add_argument('--jacquard',default=os.environ['JACQUARD_PATH'], help='path to Jacquard dataset')
    parser.add_argument('--gui',action='store_true')
    parser.add_argument('--width', default=300, type=int, help='width of image')
    parser.add_argument('--height', default=300, type=int, help='height of image')

    args = parser.parse_args()

    obj_fns = set(glob.glob(os.path.join(args.models, '*.obj')))
    vhacd_fns = set(glob.glob(os.path.join(args.models, '*_vhacd.obj')))
    obj_fns = obj_fns - vhacd_fns

    jaq = Jacquard(args.jacquard)
    sim = s.Simulator(gui=args.gui, timestep=1./240., epochs=10000)
    scales_f = open('/home/mario/Developer/msc-thesis/simulator/scales.csv', 'w')
    scales_f.write('obj_id,scale\n')
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.ion()
    for obj_fn in obj_fns:
        obj_id = obj_fn.split('/')[-1].replace('.obj','')
        scene_name = '0_' + obj_id
        try:
            data = jaq[scene_name]
            g_depth = data['perfect_depth']
            g_mask = data['mask']
        except:
            print 'Key not found, skipping'
        s_depth, s_mask = sim.cam.snap(segmentation=True)[1:3]
        plt.subplot(1,2,1)
        plt.imshow(g_depth)
        plt.title('Reference')
        g_range = depth_range(g_depth, g_mask)
        while not selected and not discarded:
            sim.reset()
            sim.load(obj_fn, ori=[roll, pitch, yaw])
            sim.run()
            s_depth, s_mask = sim.cam.snap(segmentation=True)[1:3]
            plt.subplot(1,2,2)
            plt.imshow(s_depth)
            plt.title('Simulator')
            wait_keypress()

        if selected:
            s_range = depth_range(s_depth, s_mask)
            scale = g_range/s_range
            scales_f.write('{},{}\n'.format(obj_id, scale))
            sim.reset()
            sim.load(obj_fn, scale=scale, ori=[roll, pitch, yaw])
            print 'Generating scene...'
            sim.run(epochs=int(5/1e-4), autostop=True)
            if sim.is_stable():
                save(obj_id, args.scenes, args.width, args.height)
                print 'Done.'
            else:
                print 'Unstable'
            s_depth = sim.cam.snap()[1]
            plt.imshow(s_depth)
            plt.pause(0.8)

        selected = False
        discarded = False
    scales_f.close()







