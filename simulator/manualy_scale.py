import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--models_path', default=os.environ['MODELS_PATH'])
parser.add_argument('--blender_scenes_path', default='/run/media/mario/256D52764756BC25/blender scenes')

args = parser.parse_args()
template_scene = os.path.join(args.blender_scenes_path, 'base_scene.blend')
print('Template scene: {}'.format(template_scene))
obj_fns = glob.glob(os.path.join(args.models_path, '*.obj'))


for obj_fn in obj_fns:
    os.system('blender \"{}\" --python bpy_test.py -- {}'.format(template_scene, obj_fn))
