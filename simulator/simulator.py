import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R

red = [1,0,0]
green = [0,1,0]
blue = [0,0,1]
black = [0,0,0]

class Camera:

    def __init__(self, width=300, height=300, pos=[0, 0, 2], target=np.zeros(3,)):
        self._width = width
        self._height = height
        self._target = target
        self._near = 1
        self._far = 2
        self._up = [0., 1., 0.]
        self._pos = pos
        self._view = None
        self._projection = None
        self._update_camera_parameters()

    def snap(self):
        _, _, rgb, depth, _ = p.getCameraImage(self._width, self._height, self._view, self._projection)
        depth = self._far * self._near / (self._far - (self._far - self._near) * depth)
        rgb = rgb.astype(np.uint8)
        return rgb, depth

    def _update_camera_parameters(self):
        self._view = p.computeViewMatrix(self._pos, self.target, self._up)
        self._projection = p.computeProjectionMatrixFOV(50, self.width/float(self.height), self._near, self._far)

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = pos
        self._update_camera_parameters()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, height):
        self._height = pos
        self._update_camera_properties()

    @property
    def width(self):
        return self._width

    @height.setter
    def height(self, height):
        self._height = height
        self._update_camera_properties()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._update_camera_properties()

    @property
    def left(self):
        m = np.array(self._projection).reshape(4,4).T
        return - (m[0,3] + 1)/m[0,0]

    @property
    def right(self):
        m = np.array(self._projection).reshape(4,4).T
        return 2./m[0,0] + self.left

    @property
    def bottom(self):
        m = np.array(self._projection).reshape(4,4).T
        return -(m[1,3] + 1)/m[1,1]

    @property
    def top(self):
        m = np.array(self._projection).reshape(4,4).T
        return 2./m[1,1] + self.bottom

    def world_from_camera(self, u, v, d):
        # Window to NDC transformation
        n_u = float(u)/(self.width - 1)
        n_v = (self.height - 1 - float(v))/(self.height - 1)
        # NDC to view transformation
        l,r,t,b = self.left, self.right, self.top, self.bottom
        w = r - l
        h = t - b
        c_u = l + w * n_u
        c_v = b + h * n_v
        # View to world
        p_c = np.array([c_u, c_v, -d, 1])
        view = np.array(self._view).reshape(4,4).T
        t_c_w = np.linalg.inv(view)
        world = np.dot(t_c_w, p_c)
        import ipdb; ipdb.set_trace() # BREAKPOINT
        return world[0:3]


class Simulator:

    def __init__(self, gui=False, timestep=1e-4, debug=False, epochs=10000, stop_th=1e-6, g=-10):
        self.gui = gui
        self.debug = debug
        self.epochs = epochs
        self.stop_th = stop_th
        self.timestep = timestep

        if gui:
            mode = p.GUI
        else:
            mode = p.DIRECT

        self.client = p.connect(mode)
        p.setTimeStep(self.timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(erp=0.4, contactERP=0.4, frictionERP=0.4)
        p.setGravity(0,0,g)

        self.planeId = p.loadURDF("plane.urdf")
        self.gid = None
        self.old_poses = {}
        self.obj_ids = {}
        self.cam = Camera()

    @property
    def bodies(self):
        for i in range(p.getNumBodies()):
            bid = p.getBodyUniqueId(i)
            if bid != self.planeId and bid != self.gid:
                yield bid

    def load(self, fn, pos=None, ori=None):
        extension = fn.split('.')[-1].lower()
        assert extension == 'urdf' or extension == 'obj'

        if extension == 'urdf':
            if pos is None:
                pos = [0,0,0]
            bid = p.loadURDF(fn, basePosition=pos)
        else:
            bid = self._load_obj(fn, pos, ori)

        self.old_poses[bid] = self._get_pose(bid)
        self.obj_ids[bid] = fn.split('/')[-1].split('.')[-2]

        return bid

    def _load_obj(self, fn, pos, ori):
        visual_fn = fn
        collision_fn = fn.split('.')[-2] + '_vhacd.obj'
        visual_fn = collision_fn # for debuggin purposes
        obj_id = fn.split('/')[-1].split('.')[-2]

        if not os.path.exists(collision_fn):
            print('''No collision model for {} was found. Falling back to
                    visual model as collision geometry. This might impact
                    performance of your simulation''').format(visual_fn)
            collision_fn = visual_fn

        root = fn.split('/')[-2]

        metadata_fn = os.path.join(root, 'metadata.csv')

        scale = 1
        mass = 1

        if os.path.isfile(metadata_fn):
            csv = pd.read_csv(metadata_fn).set_index(u'fullId')
            ids = csv.index
            wss_id = u'wss.'+obj_id

            id_found = wss_id in ids

            if id_found:
                scale = csv.loc[wss_id][u'unit']
                # Check if NaN (due to missing value)
                if scale != scale:
                    scale = 1
                mass = csv.loc[wss_id][u'weight']
                if mass != mass:
                    mass = 1

        scale = np.repeat(scale, 3)

        # First load obj to get aabb
        vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn, rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale)
        cId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_fn, meshScale=scale)
        bId = p.createMultiBody(baseMass=mass,baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId)
        aabb = np.array(p.getAABB(bId))
        p.removeBody(bId)

        # Compute size and reload with proper shift
        size = np.abs(aabb[1] - aabb[0])
        shift = -aabb[0] - size/2.
        shift[2] = -size[2]/2.

        if pos is None:
            start_pos = np.zeros((3,))
            start_pos[2] = np.max(size)
        else:
            start_pos = pos

        if ori is None:
            start_ori = np.random.uniform(0, 2*np.pi, (3,)).tolist()
            start_ori = p.getQuaternionFromEuler(start_ori)
        else:
            start_ori = p.getQuaternionFromEuler(ori)

        vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn, rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale, visualFramePosition=shift)
        cId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_fn, meshScale=scale, collisionFramePosition=shift)
        bId =  p.createMultiBody(baseMass=1,baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId, basePosition=start_pos, baseOrientation=start_ori)
        #p.changeDynamics(bId, -1, lateralFriction=1, contactStiffness=1, contactDamping=1)
        p.changeDynamics(bId, -1, lateralFriction=1)
        #info = p.getDynamicsInfo(bId, -1)

        return bId

    def add_gripper(self, gripper_fn):
        self.gid = p.loadURDF(gripper_fn, basePosition=[0, 0, 0.06])
        #info = [p.getDynamicsInfo(self.gid, l) for l in range(p.getNumJoints(self.gid))]
        #p.changeDynamics(self.gid, -1, contactStiffness=1000, contactDamping=10000)

    def _remove_body(self, bId):
        del self.old_poses[bId]
        p.removeBody(bId)

    def reset(self):
        for bId in self.bodies:
            p.removeBody(bId)
        self.old_poses.clear()

    def is_stable(self):
        for k, v in self.old_poses.iteritems():
            old_pos, old_ori = v
            pos, ori = self._get_pose(k)
            if not (np.linalg.norm(np.abs(pos - old_pos)) < self.stop_th and
                    np.linalg.norm(np.abs(ori - old_ori)) < self.stop_th):
                return False
        return True

    def save(self, fn):
        """
            Format of saved file: obj_id,pos(x3),ori(x3)
            This will save a csv file with the shapenet object id and its pose,
            along with a bullet file with the dynamic state of each object
        """
        with open(fn, mode='w') as csv:
            csv.write('obj_id,x,y,z,a,b,g\n')
            for bid in self.bodies:
                obj_id = self.obj_ids[bid]
                pos, ori = p.getBasePositionAndOrientation(bid)
                ori = p.getEulerFromQuaternion(ori)
                obj_str = ','.join([obj_id] + [str(ps) for ps in pos] + [ str(o) for o in ori]) + '\n'
                csv.write(obj_str)
        p.saveBullet(fn.split('.')[-2] + '.bullet')

    def restore(self, fn, path):
        """
            Will read the .world and .bullet files associated to {fn} and load
            the world accordingly. Will look for linked .obj files in the
            {path} directory
        """
        assert os.path.exists(fn)

        self.reset()

        with open(fn) as csv:
            csv.readline() # Skip descriptor line
            for line in csv:
                fields = line.split(',')
                obj_fn = os.path.join(path, fields[0] + '.obj')
                obj_pos = [float(fields[1]),float(fields[2]),float(fields[3])]
                obj_ori = [float(fields[4]),float(fields[5]),float(fields[6])]
                self.load(obj_fn, obj_pos, obj_ori)

        p.restoreState(fileName=fn.split('.')[-2] + '.bullet')

    def _update_pos(self):
        for id in self.bodies:
            self.old_poses[id] = self._get_pose(id)

    def _get_pose(self, bId):
        pos, ori = p.getBasePositionAndOrientation(bId)
        return np.array(pos), np.array(ori)

    def drawFrame(self, frame, parent=-1):
        f = np.tile(np.array(frame), (3,1))
        t = f + np.eye(3)

        p.addUserDebugLine(f[0,:], t[0,:], red, parentObjectUniqueId=parent)
        p.addUserDebugLine(f[1,:], t[1,:], green, parentObjectUniqueId=parent)
        p.addUserDebugLine(f[2,:], t[2,:], blue, parentObjectUniqueId=parent)

    def close_gripper(self):
        for joint in [6,7]:
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=0.1, maxVelocity=0.1)
        self.run_action(tolerance=1e-2)

    def open_gripper(self):
        for joint in [6,7]:
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=0, maxVelocity=1)
        self.run_action()

    def move_gripper_to(self, pose):
        for joint in range(6):
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=pose[joint], maxVelocity=1)
        self.run_action()

    def drawAABB(self, bb, parent=-1, color=black):
        bb = np.array(bb)
        if parent != -1:
            pos, ori = self._get_pose(parent)
            r = p.getMatrixFromQuaternion(ori)
            # Inverse of the rotation
            r = R.from_quat(ori).as_dcm().T
            # Transpose bb to make multiplication feasible
            bb = bb.T
            # rotate
            bb = np.dot(r, bb).T
            # translate
            bb = bb + np.vstack((pos, pos))
        x,y,z = np.abs(np.diag(bb[1] - bb[0]))
        o = bb[0]

        # Bottom face
        p.addUserDebugLine(o, o + x, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o, o + y, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+x, o + x + y, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+y, o + x + y, color, parentObjectUniqueId=parent)
        # Top face
        p.addUserDebugLine(o+z, o + x+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+z, o + y+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+x+z, o + x + y+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+y+z, o + x + y+z, color, parentObjectUniqueId=parent)
        # Sides
        p.addUserDebugLine(o, o+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+x, o+x+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+y, o+y+z, color, parentObjectUniqueId=parent)
        p.addUserDebugLine(o+y+x, o+y+x+z, color, parentObjectUniqueId=parent)

    def debug_viz(self):
        if self.debug:
            for bid in self.bodies:
                self.drawFrame([0,0,0], bid)

    def run(self, epochs=None, autostop=True):
        if epochs is None:
            epochs = self.epochs
        self.debug_viz()

        for i in range(epochs):
            p.stepSimulation()
            time.sleep(self.timestep)
            if autostop and self.is_stable():
                break

            self._update_pos()

    def debug_run(self):
        p.removeAllUserDebugItems()
        p.addUserDebugParameter("x",-2,2,0)
        p.addUserDebugParameter("y",-2,2,0)
        p.addUserDebugParameter("z",-5,0,0)
        p.addUserDebugParameter("roll",-1.5,1.5,0)
        p.addUserDebugParameter("pitch",-1.5,1.5,0)
        p.addUserDebugParameter("yaw",-1.5,1.5,0)
        p.addUserDebugParameter("width",0,0.1,0)
        old_values = [0]*7
        while True:
            p.stepSimulation()
            time.sleep(self.timestep)
            values = [p.readUserDebugParameter(id) for id in range(7)]
            values[6] /= 2.
            values.append(-values[6])
            p.setJointMotorControlArray(bodyUniqueId=self.gid, jointIndices=range(8),
                    controlMode=p.POSITION_CONTROL, targetPositions=values)


    def run_action(self, tolerance=1e-2):
        """
            Runs previously commanded action until manipulator joints stop
        """
        # Step first to get some motion
        p.stepSimulation()
        time.sleep(self.timestep)

        n_joints = p.getNumJoints(self.gid)
        states = p.getJointStates(self.gid, range(n_joints))
        velocities = [j[1] for j in states]

        while (np.abs(np.array(velocities)) > tolerance).any():
            p.stepSimulation()
            time.sleep(self.timestep)

            states = p.getJointStates(self.gid, range(n_joints))
            velocities = [j[1] for j in states]

