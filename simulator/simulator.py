import pybullet as p
import struct
import pybullet_data
import numpy as np
import os
import time
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R

red = [1,0,0]
green = [0,1,0]
blue = [0,0,1]
black = [0,0,0]


class Camera:

    def __init__(self, width=300, height=300, pos=[0, 0, 2], target=np.zeros(3,), debug=False):
        self._width = width
        self._height = height
        self._target = target
        self._near = 1
        self._far = 2
        self._up = [0., 1., 0.]
        self._pos = pos
        self.debug = debug
        self._view = None
        self._projection = None
        self._reproject = None
        self._update_camera_parameters()

    def _compute_depth(self, depth):
        normal = np.array(self._target) - np.array(self._pos)
        for u in range(300):
            for v in range(300):
                pixel = np.array([2.*u/300. - 1, 2.*v/300. - 1, -1., 1])
                point = np.dot(self._reproject, pixel)[0:3]
                point /= np.linalg.norm(point)
                depth[v,u] /= np.dot(normal, point)
        return cos

    def snap2(self):
        _, _, rgb, depth, _ = p.getCameraImage(self._width, self._height, self._view, self._projection)
        depth = self._far * self._near / (self._far - (self._far - self._near) * depth)
        rgb = rgb.astype(np.uint8)
        pos = np.array(self._pos)
        normal = np.array(self._target) - pos
        normal /= np.linalg.norm(normal)
        for u in range(self.width):
            for v in range(self.height):
                pixel = np.array([2.*u/self.width - 1, 2.*v/self.height - 1, -1., 1])
                point = np.dot(self._reproject, pixel)[0:3] - pos
                point /= np.linalg.norm(point)
                depth[v,u] /= np.dot(normal, point)

        return rgb, depth

    def snap(self):
        _, _, rgb, depth, _ = p.getCameraImage(self._width, self._height, self._view, self._projection)
        depth = self._far * self._near / (self._far - (self._far - self._near) * depth)
        rgb = rgb.astype(np.uint8)
        return rgb, depth

    def _update_camera_parameters(self):
        self._view = p.computeViewMatrix(self._pos, self.target, self._up)
        self._projection = p.computeProjectionMatrixFOV(50, self.width/float(self.height), self._near, self._far)
        v = np.array(self._view).reshape(4,4).T
        pr = np.array(self._projection).reshape(4,4).T
        self._reproject = np.dot(np.linalg.inv(v), np.linalg.inv(pr))

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

    def world_from_camera2(self, u, v, d):
        v = self.height - v
        # Window to NDC transformation
        n_u = 2.*u/self.width - 1
        n_v = 2.*v/self.height - 1
        ndc = np.array([n_u, n_v, -self._near, 1.])
        # NDC to view
        to = np.dot(self._reproject, ndc)[0:3]
        fr = np.array(self._pos)
        ray = to-fr
        ray /= np.linalg.norm(ray)
        tgt = np.array(self.target) - np.array(self.pos)
        #d = d * np.linalg.norm(tgt) / np.dot(ray, tgt)
        world = fr + ray*d

        if self.debug:
            p.addUserDebugLine(fr, fr + 3*(cn[:3,1]-fr)/np.linalg.norm(cn[:3,1]-fr))
            p.addUserDebugLine(fr, world[0:3])


        return world[0:3]

    def world_from_camera(self, u, v, d):
        v = self.height - v
        # Window to NDC transformation
        n_u = 2.*u/self.width - 1
        n_v = 2.*v/self.height - 1
        ndc = np.array([n_u, n_v, -self._near, 1.])
        # NDC to world
        to = np.dot(self._reproject, ndc)[0:3]
        fr = np.array(self._pos)
        ray = to-fr
        ray /= np.linalg.norm(ray)
        tgt = np.array(self.target) - np.array(self.pos)
        d = d * np.linalg.norm(tgt) / np.dot(ray, tgt)
        world = fr + ray*d

        if self.debug:
            p.addUserDebugLine(fr, fr + 3*(cn[:3,1]-fr)/np.linalg.norm(cn[:3,1]-fr))
            p.addUserDebugLine(fr, world[0:3])


        return world[0:3]

    def show_frustrum(self):
        if self.debug:
            #column order ul, ur, dl, dr
            frame = np.array([[-1, 1, -1, 1], [1,1,-1,1], [-1,-1,-1,1], [1,-1,-1,1]]).T
            cn = np.dot(self._reproject, frame)
            p.addUserDebugLine(cn[:3,0],cn[:3,1])
            p.addUserDebugLine(cn[:3,0],cn[:3,2])
            p.addUserDebugLine(cn[:3,2],cn[:3,3])
            p.addUserDebugLine(cn[:3,3],cn[:3,1])

    def compute_grasp(self, bb, depth):
        """
        Computes the center world coordinate and width size of a grasp defined
        by its bounding box. bb is a (4,2) np array with the corners of the
        grasp
        """
        center_pixels = np.mean(bb, axis=0).astype(np.int)
        center_world = self.world_from_camera(center_pixels[1], center_pixels[0], depth)
        corners_world = np.zeros((3,4))
        # Width line
        p1 = self.world_from_camera(bb[1,1], bb[1,0], depth)
        p0 = self.world_from_camera(bb[0,1], bb[0,0], depth)
        width = np.linalg.norm(p1-p0)

        return center_world, width


class Simulator:

    def __init__(self, gui=False, timeout=60, timestep=1e-4, debug=False,
            epochs=10000, stop_th=1e-6, g=-10, bin_pos=[1.5, 1.5, 0.01]):
        self.gui = gui
        self.debug = debug
        self.epochs = epochs
        self.stop_th = stop_th
        self.timestep = timestep
        self.timeout = timeout
        self.bin_pos = bin_pos

        if gui:
            mode = p.GUI
        else:
            mode = p.DIRECT

        self.client = p.connect(mode)
        p.setTimeStep(self.timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.setPhysicsEngineParameter(erp=0.4, contactERP=0.4, frictionERP=0.4)
        p.setGravity(0,0,g)

        self.planeId = p.loadURDF("plane.urdf")
        self.gid = None
        self.bin = None
        self.old_poses = {}
        self.obj_ids = {}
        self.cam = Camera()

    @property
    def bodies(self):
        for i in range(p.getNumBodies()):
            bid = p.getBodyUniqueId(i)
            if bid != self.planeId and bid != self.gid and bid != self.bin:
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

    def obj_center(self, obj_fn):
        vertices = []
        with open(obj_fn, 'r') as f:
            for line in f.readlines():
                fields = line.split(' ')
                if fields[0] == 'v':
                    vertices.append([float(x) for x in fields[1:4]])

        vertices = np.array(vertices)
        return vertices.mean(axis=0)

    def _load_obj(self, fn, pos, ori):
        visual_fn = fn
        collision_fn = fn.split('.')[-2] + '_vhacd.obj'
        visual_fn = collision_fn # Temporary fix until precise vhacd decomposition achieved
        obj_id = fn.split('/')[-1].split('.')[-2]

        if not os.path.exists(collision_fn):
            print('''No collision model for {} was found. Falling back to
                    visual model as collision geometry. This might impact
                    performance of your simulation''').format(visual_fn)
            collision_fn = visual_fn

        root = '/'.join(fn.split('/')[:-1])

        metadata_fn = os.path.join(root, 'metadata.csv')

        scale = 1
        #mass = 1

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
                #mass = csv.loc[wss_id][u'weight']
                #if mass != mass:
                #    mass = 1

        scale = np.repeat(scale, 3).astype(np.float)

        # First load obj to get aabb
        vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn, rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale)
        cId = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_fn, meshScale=scale)
        bId = p.createMultiBody(baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId)
        aabb = np.array(p.getAABB(bId))
        p.removeBody(bId)

        # Compute size and reload with proper shift
        size = np.abs(aabb[1] - aabb[0])
        max_dim = np.argmax(size)
        new_scale = np.clip(size[max_dim], 0.08, 0.9)/size[max_dim]
        size *= new_scale
        scale *= np.repeat(new_scale, 3)

        center = self.obj_center(visual_fn)
        center *= scale


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

        vId = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn,
                rgbaColor=[1,1,1,1], specularColor=[0.4,.4,0], meshScale=scale,
                visualFramePosition=-center)
        cId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                fileName=collision_fn, meshScale=scale,
                collisionFramePosition=-center)
        bId =  p.createMultiBody(baseMass=size[max_dim],baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cId, baseVisualShapeIndex = vId, basePosition=start_pos, baseOrientation=start_ori)
        if self.debug:
            self.drawFrame([0,0,0], bId)

        return bId

    def replay(self, log_fn, scene_fn):
        self.restore(scene_fn, os.environ['SHAPENET_PATH'])
        self.add_gripper('/Users/mario/Developer/msc-thesis/simulator/gripper.urdf')
        log = self._read_logfile(log_fn)

        for record in log:
            Id = record[2]
            pos = [record[3],record[4],record[5]]
            orn = [record[6],record[7],record[8],record[9]]
            p.resetBasePositionAndOrientation(Id,pos,orn)
            numJoints = p.getNumJoints(Id)
            for i in range (numJoints):
                jointInfo = p.getJointInfo(Id,i)
                qIndex = jointInfo[3]
                if qIndex > -1:
                    p.resetJointState(Id,i,record[qIndex-7+17])
            #time.sleep(self.timestep)

    def _read_logfile(self, filename, verbose = True):
        f = open(filename, 'rb')

        print('Opened'),
        print(filename)

        keys = f.readline().decode('utf8').rstrip('\n').split(',')
        fmt = f.readline().decode('utf8').rstrip('\n')

        # The byte number of one record
        sz = struct.calcsize(fmt)
        # The type number of one record
        ncols = len(fmt)

        if verbose:
            print('Keys:'),
            print(keys)
            print('Format:'),
            print(fmt)
            print('Size:'),
            print(sz)
            print('Columns:'),
            print(ncols)

        # Read data
        wholeFile = f.read()
        # split by alignment word
        chunks = wholeFile.split(b'\xaa\xbb')
        log = list()
        for chunk in chunks:
            if len(chunk) == sz:
                values = struct.unpack(fmt, chunk)
                record = list()
                for i in range(ncols):
                    record.append(values[i])
                log.append(record)

        return log

    def add_gripper(self, gripper_fn):
        self.gid = p.loadURDF(gripper_fn)
        self.drawFrame([0,0,0])
        p.resetJointState(self.gid, 2, 1)

    def evaluate_grasp(self, pose, width, log_fn=None):
        if log_fn:
            log = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, log_fn)
        if not self.gid:
            self.add_gripper('simulator/gripper.urdf')
        #if not self.bin:
        #    self.bin = p.loadURDF('simulator/bin.urdf', basePosition=self.bin_pos)
        self.open_gripper()
        # Move to pregrasp
        self.move_gripper_to(pose + np.array([0, 0, 0.1, 0, 0, 0]))
        self.set_gripper_width(width)
        # Move to grasp
        self.move_gripper_to(pose)
        self.close_gripper()
        # Move to postgrasp
        #self.move_gripper_to(pose + np.array([0, 0, 1.5, 0, 0, 0]))
        self.move_gripper_to(np.array([0, 0, 1.5, 0, 0, 0]))
        bid = self.bodies.next()
        final_pos,_ = p.getBasePositionAndOrientation(bid)
        print('Final pos: ', final_pos)
        if log_fn:
            p.stopStateLogging(log)
        return np.abs(final_pos[2] - 1.5) < 0.5

        # Drop object in bin
        self.move_gripper_to(np.array([self.bin_pos[0], self.bin_pos[1], 1.2, 0, 0, pose[5]]))
        self.open_gripper()
        self.run(epochs=int(4./self.timestep), autostop=False)
        bid = self.bodies.next()
        final_pos,_ = p.getBasePositionAndOrientation(bid)
        print('Final pos: ', final_pos)
        if log_fn:
            p.stopStateLogging(log)
        return np.linalg.norm(np.array(final_pos[0:2]) - np.array(self.bin_pos[0:2])) < 0.7

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
            Will read the .csv and .bullet files associated to {fn} and load
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
                print('Loaded '+ obj_fn)

        #p.restoreState(fileName=fn.split('.')[-2] + '.bullet')

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
        self.set_gripper_width(0)

    def open_gripper(self):
        self.set_gripper_width(0.1)

    def set_gripper_width(self, width):
        width = 0.1 - width
        for joint in [6,7]:
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=width/2., maxVelocity=1)
        self.run(epochs=int(0.1/self.timestep))

    def move_gripper_to(self, pose, pos_tol=0.01, ang_tol=2):
        ang_tol = ang_tol*np.pi/180.
        pose[2] += 0.02
        for joint in range(3):
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=pose[joint], maxVelocity=2)
        for joint in [3, 4, 5]:
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=pose[joint], maxVelocity=15)

        for _ in range(int(self.timeout/self.timestep)):
            state = p.getLinkState(self.gid, 5)
            pos = np.array(state[0])
            ori = np.array(p.getEulerFromQuaternion(state[1]))
            ori = np.array([x[0] for x in p.getJointStates(self.gid, [3,4,5])])
            if np.linalg.norm(pos - pose[0:3]) < pos_tol and (np.abs(ori - pose[3:6]) < ang_tol).all():
                print('Arrived within tolerance')
                break
            p.stepSimulation()
            self.sleep()
        else:
            print('Move timed out')

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

    def sleep(self):
        if self.gui:
            time.sleep(self.timestep)

    def debug_viz(self):
        if self.debug:
            for bid in self.bodies:
                self.drawFrame([0,0,0], bid)

    def run(self, epochs=None, autostop=False):
        if epochs is None:
            epochs = self.epochs
        self.debug_viz()

        for i in range(epochs):
            p.stepSimulation()
            self.sleep()
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
            self.sleep()
            values = [p.readUserDebugParameter(id) for id in range(7)]
            values[6] /= 2.
            values.append(-values[6])
            p.setJointMotorControlArray(bodyUniqueId=self.gid, jointIndices=range(8),
                    controlMode=p.POSITION_CONTROL, targetPositions=values)

    def run_action(self, tolerance=1e-2):
        p.stepSimulation()
        self.sleep()
        velocities = lambda: [joint[1] for joint in p.getJointStates(self.gid, range(8))]

        for _ in range(int(self.timeout/self.timestep)):
            if (np.abs(np.array(velocities())) < tolerance).all():
                break
            p.stepSimulation()
            self.sleep()
        else:
            print('Action timed out')

    def wait_move(self, goal, pos_tol=1e-3, ang_tol=1e-2):
        p.stepSimulation()
        self.sleep()
        pos = lambda: np.array(p.getLinkState(self.gid, 5)[0])
        ori = lambda: np.array(p.getEulerFromQuaternion(p.getLinkState(self.gid, 5)[1]))
        goal = np.array(goal)

        for _ in range(int(self.timeout/self.timestep)):
            import ipdb; ipdb.set_trace() # BREAKPOINT
            if (np.abs(pos() - goal[0:3]) < pos_tol).all() and np.min(np.vstack((np.abs(goal[3:6] - ori()), np.abs(ori() - goal[3:6]))), axis=0):
                pass
        pass

