import pybullet as p
import glob
import argparse
import os
import time
import pybullet_data
import numpy as np

STOP_TH = 1e-7

red = [1,0,0]
green = [0,1,0]
blue = [0,0,1]
black = [0,0,0]

def drawFrame(frame, parent):

    f = np.tile(np.array(frame), (3,1))
    t = f + np.eye(3)

    p.addUserDebugLine(f[0,:], t[0,:], red, parentObjectUniqueId=parent)
    p.addUserDebugLine(f[1,:], t[1,:], green, parentObjectUniqueId=parent)
    p.addUserDebugLine(f[2,:], t[2,:], blue, parentObjectUniqueId=parent)

def drawAABB(bb, parent):
    bb = np.array(bb)
    x,y,z = np.abs(np.diag(bb[1] - bb[0]))
    o = bb[0]

    # Bottom face
    p.addUserDebugLine(o, o + x, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o, o + y, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+x, o + x + y, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+y, o + x + y, black, parentObjectUniqueId=parent)
    # Top face
    p.addUserDebugLine(o+z, o + x+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+z, o + y+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+x+z, o + x + y+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+y+z, o + x + y+z, black, parentObjectUniqueId=parent)
    # Sides
    p.addUserDebugLine(o, o+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+x, o+x+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+y, o+y+z, black, parentObjectUniqueId=parent)
    p.addUserDebugLine(o+y+x, o+y+x+z, black, parentObjectUniqueId=parent)

def loadOBJ(obj_path, obj_name, start_pos=np.zeros((3,)), start_ori=np.zeros((3,))):
    visual_fn = os.path.join(obj_path,'{}.obj').format(obj_name)
    collision_fn = os.path.join(obj_path,'{}_vhacd.obj').format(obj_name)
    scale = np.repeat(0.01, 3)

    vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn, rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale)
    cId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_fn, meshScale=scale)
    bId = p.createMultiBody(baseMass=1,baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId)
    aabb = np.array(p.getAABB(bId))
    p.removeBody(bId)

    size = np.abs(aabb[1] - aabb[0])
    shift = -aabb[0] - size/2.
    shift[2] = -size[2]/2.
    start_pos[2] = np.max([start_pos[2], np.max(size)])

    vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn, rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale, visualFramePosition=shift)
    cId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_fn, meshScale=scale, collisionFramePosition=shift)
    return p.createMultiBody(baseMass=1,baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId, basePosition=start_pos, baseOrientation=start_ori)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to .obj')

    args = parser.parse_args()

    obj_fns = glob.glob(os.path.join(args.path, '*_vhacd.obj'))
    assert(len(obj_fns) > 0)

    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    p.setGravity(0,0,-10)

    planeId = p.loadURDF("plane.urdf")

    start_pos = [0,0,1]

    for obj_fn in obj_fns:

        obj_name = obj_fn.split('/')[-1].split('_vhacd.obj')[0]

        start_ori = p.getQuaternionFromEuler(np.random.uniform(0, 2*np.pi, (3,)))

        mbId = loadOBJ(args.path, obj_name, start_pos, start_ori)

        old_pos, old_ori = p.getBasePositionAndOrientation(mbId)
        old_pos = np.array(old_pos)
        old_ori = np.array(old_ori)


        for i in range (10000):
            p.stepSimulation()
            time.sleep(1./240.)

            pos, ori = p.getBasePositionAndOrientation(mbId)
            pos = np.array(pos)
            ori = np.array(ori)

            if np.linalg.norm(np.abs(pos - old_pos)) < STOP_TH and np.linalg.norm(np.abs(ori - old_ori)) < STOP_TH and i:
                print('\n===Stabilized===\n')
                p.saveWorld(obj_name + '_scene.bullet')
                break

            old_pos = pos
            old_ori = ori
        else:
            print('\n===Simulation did not become stable after 10000 iterations===\n')

        p.removeBody(mbId)

    p.disconnect()


