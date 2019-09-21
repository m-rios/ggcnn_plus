import bpy
import sys

BLENDER_SCENES_PATH='/Users/mario/Developer/msc-thesis/data/3d_models/blender_scenes'
obj_fn = sys.argv[-1]
obj_id = obj_fn.split('/')[-1].replace('.obj', '')
bpy.ops.import_scene.obj(filepath=obj_fn)
for ob in bpy.context.scene.objects:
    if ob.data.name != 'Plane' and ob.data.name != 'Cube':
        bpy.data.objects[ob.data.name].select = True
        bpy.ops.object.origin_set( type = 'GEOMETRY_ORIGIN' )
bpy.ops.wm.save_as_mainfile(filepath=BLENDER_SCENES_PATH + '/' + obj_id + '.blend')
