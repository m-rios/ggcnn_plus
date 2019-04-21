import bpy
import glob
import os

BLENDER_SCENES_PATH='/home/mario/data/blender scenes'
blender_fns = glob.glob(os.path.join(BLENDER_SCENES_PATH, '*.blend'))
scales_f = open('/home/mario/Developer/msc-thesis/simulator/scales.csv','w')
scales_f.write('obj_id,scale\n')
for blender_fn in blender_fns:
    obj_id = blender_fn.split('/')[-1].replace('.blend', '')

    bpy.ops.wm.open_mainfile(filepath=blender_fn)
    bpy.ops.object.select_all( action = 'SELECT' )
    bpy.ops.object.origin_set( type = 'ORIGIN_GEOMETRY' )

    for ob in bpy.context.scene.objects:
        if ob.data.name != 'Plane' and ob.data.name != 'Cube':
            scales_f.write('{},{}\n'.format(obj_id, abs(ob.scale[0])))
