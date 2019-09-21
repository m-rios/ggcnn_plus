import os
import glob
import argparse
import shutil
BLENDER = '/Applications/Blender/blender.app/Contents/MacOS/blender'
parser = argparse.ArgumentParser()
parser.add_argument('--models_path', default=os.environ['MODELS_PATH'])
parser.add_argument('--blender_scenes_path', default='/Users/mario/Developer/msc-thesis/data/3d_models/blender_scenes')

args = parser.parse_args()

template_scene = os.path.join(args.blender_scenes_path, 'base_scene.blend')
print('Template scene: {}'.format(template_scene))
obj_fns = glob.glob(os.path.join(args.models_path, '*.obj'))


for obj_fn in obj_fns:
    new_scene = obj_fn.split('/')[-1].replace('.obj', '')
    new_scene = template_scene.replace('base_scene', new_scene)
    shutil.copyfile(template_scene, new_scene)
    os.system('{} \"{}\" --python bpy_test.py -- {}'.format(BLENDER, new_scene, obj_fn))
