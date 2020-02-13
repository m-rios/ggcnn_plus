import pybullet as p
import skvideo.io
import cv2
import struct
import pybullet_data
import numpy as np
import os
import time
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R
from utils import Wavefront, silence_stdout, auto_str
import pkgutil
egl = pkgutil.get_loader('eglRenderer')

red = [1,0,0]
green = [0,1,0]
blue = [0,0,1]
black = [0,0,0]

VIDEO_LOGGER = 0
STATE_LOGGER = 1
OPENGL_LOGGER = 2
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))


class VideoLogger(object):
    def __init__(self, log_fn, timestep, rate=2.0, shape=(900, 900), pos=[-0.75, -0.75, 2]):
        self.video = skvideo.io.FFmpegWriter(log_fn,outputdict={'-vcodec': 'libx264'})
        self.rate = rate
        self.fn = log_fn
        self.cam = Camera(width=shape[0], height=shape[1], pos=pos, target=[0.5, 0.5,0], far=6,up=[1., 1., 0.], fov=60)
        self.timestep = timestep
        self.epoch = 0
        self.shape = shape

    def log(self):
        if self.epoch % (1./self.timestep//self.rate) == 0:
            rgb = self.cam.snap()[0]
            rgb = rgb[:,:,0:3].astype(np.uint8).reshape(self.shape[0],self.shape[1],3)
            self.video.writeFrame(rgb)
        self.epoch += 1

    def close(self):
        self.video.close()

class StateLogger(object):
    def __init__(self, log_fn):
        self.log_fn = log_fn
        self.logid = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, self.log_fn)

    def log(self):
        pass

    def close(self):
        p.stopStateLogging(self.logid)

class OpenGLLogger(object):
    """
        NOTE: IMPRACTICALLY SLOW, DO NO USE
    """
    def __init__(self, log_fn):
        self.log_fn = log_fn
        self.logid = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.log_fn)

    def log(self):
        pass

    def close(self):
        p.stopLogging(self.logid)


@auto_str
class Camera(object):

    def __init__(self, width=300, height=300, fov=40, pos=[0, 0, 1.5],
            target=np.zeros(3,),far=5, near=0.1, up=[0., 1., 0],  debug=False):
        self._width = width
        self._height = height
        self._target = target
        self._near = near
        self._far =  far
        self._up = up
        self._pos = pos
        self.debug = debug
        self._view = None
        self._projection = None
        self._reproject = None
        self._fov = fov
        self._update_camera_parameters()

    def _compute_depth(self, depth):
        normal = np.array(self._target) - np.array(self._pos)
        for u in range(self.width):
            for v in range(self.height):
                pixel = np.array([2.*u/self.width - 1, 2.*v/self.height - 1, -1., 1])
                point = np.dot(self._reproject, pixel)[0:3]
                point /= np.linalg.norm(point)
                depth[v,u] /= np.dot(normal, point)
        return cos

    def snap(self, segmentation=False):
        _, _, rgb, depth, mask = p.getCameraImage(self._width, self._height, self._view, self._projection)
        rgb = np.array(rgb).reshape(self.height, self.width, 4)
        depth = np.array(depth).reshape(self.height, self.width)
        mask = np.array(mask).reshape(self.height, self.width).astype(np.bool)
        depth = self._far * self._near / (self._far - (self._far - self._near) * depth)
        rgb = rgb.astype(np.uint8)
        if segmentation:
            return rgb, depth, mask
        else:
            return rgb, depth

    def point_cloud(self):
        """
        Returns an array of 3D points representing the point cloud extracted from the depth image.
        :return: A numpy array with 3d points
        """
        _, depth = self.snap()
        pcd = []

        for r in range(depth.shape[0]):
            for c in range(depth.shape[1]):
                pcd.append(self.world_from_camera(c, r, depth[r, c]))

        return np.array(pcd)

    def depth_buffer(self):
        return p.getCameraImage(self._width, self._height, self._view, self._projection)[3]

    def _update_camera_parameters(self):
        self._view = p.computeViewMatrix(self._pos, self._target, self._up)
        self._projection = p.computeProjectionMatrixFOV(self._fov, self.width/float(self.height), self._near, self._far)
        v = np.array(self._view).reshape(4,4).T
        pr = np.array(self._projection).reshape(4,4).T
        self._reproject = np.linalg.inv(np.dot(pr, v))

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
        self._height = height
        self._update_camera_parameters()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, width):
        self._width = width
        self._update_camera_parameters()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target
        self._update_camera_parameters()

    def depth_to_buffer(self, d):
        f = float(self._far)
        n = float(self._near)
        return f/(f-n) - f*n/((f-n) * d)

    def world_from_camera(self, u, v, d):
        v = self.height - v
        # Window to NDC transformation
        n_u = 2.*u/self.width - 1
        n_v = 2.*v/self.height - 1
        buffer_d = self.depth_to_buffer(d)
        n_z = (2.*buffer_d - self._far - self._near)/(self._far - self._near)
        n_z = buffer_d * 2. - 1.
        ndc = np.array([n_u, n_v, -self._near, 1.])
        ndc = np.array([n_u, n_v, n_z, 1.])
        # NDC to world
        to = np.dot(self._reproject, ndc)[0:4]
        return to[0:3]/to[3]

    def project(self, pt):
        pt = np.array(pt + [1]).astype(np.float32)
        v = np.array(self._view).reshape(4,4).T
        pr = np.array(self._projection).reshape(4,4).T
        pt = np.dot(v, pt)
        pt = np.dot(pr, pt)
        pt /= pt[3]
        pt = pt[0:3]
        pt *= [self.width/2., self.height/2., (self._far - self._near)/2.]
        pt += [self.width/2., self.height/2., (self._near + self._far)/2.]

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
        p0 = self.world_from_camera(bb[0,1], bb[0,0], depth)
        p1 = self.world_from_camera(bb[1,1], bb[1,0], depth)
        p2 = self.world_from_camera(bb[2,1], bb[2,0], depth)
        p3 = self.world_from_camera(bb[3,1], bb[3,0], depth)
        width = np.linalg.norm(p1-p0)

        if self.debug:
            p.removeAllUserDebugItems()
            p.addUserDebugLine(self.pos, center_world)
            p.addUserDebugLine(p0, p1, [0, 0, 0])
            p.addUserDebugLine(p1, p2, [0, 1, 0])
            p.addUserDebugLine(p2, p3, [0, 0, 0])
            p.addUserDebugLine(p3, p0, [0, 1, 0])

        return center_world, width


@auto_str
class Simulator:

    def __init__(self, gui=False, use_egl=True, timeout=2, timestep=1e-3, debug=False,
            epochs=10000, stop_th=1e-4, g=-10, bin_pos=[1.5, 1.5, 0.01], visual_model=False):
        self.gui = gui
        self.debug = debug
        self.epochs = epochs
        self.epoch = 0
        self.stop_th = stop_th
        self.timestep = timestep
        self.timeout = timeout
        self.bin_pos = bin_pos
        self.visual_model = visual_model

        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
            if use_egl:
                plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.setTimeStep(self.timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.setPhysicsEngineParameter(erp=0.4, contactERP=0.4, frictionERP=0.4)
        p.setGravity(0,0,g)

        self.planeId = p.loadURDF("plane.urdf")
        self.gid = None
        self.bin = None
        self.old_poses = {}
        self.obj_ids = {}
        self.cam = Camera(debug=debug)
        self.logger = None

        self.x_slider = None
        self.y_slider = None
        self.z_slider = None

        self.r_slider = None
        self.p_slider = None
        self.yaw_slider = None

    def __del__(self):
        p.disconnect()

    @property
    def bodies(self):
        for i in range(p.getNumBodies()):
            bid = p.getBodyUniqueId(i)
            if bid != self.planeId and bid != self.gid and bid != self.bin:
                yield bid

    def load(self, fn, pos=None, ori=None, scale=1):
        extension = fn.split('.')[-1].lower()
        assert extension == 'urdf' or extension == 'obj'

        if extension == 'urdf':
            if pos is None:
                pos = [0,0,0]
            bid = p.loadURDF(fn, basePosition=pos)
        else:
            #bid = self._load_obj(fn, pos, ori)
            bid = self.load_scale(fn, pos, ori, scale)


        self.old_poses[bid] = self._get_pos_and_ori(bid)
        self.obj_ids[bid] = fn.split('/')[-1].split('.')[-2]

        return bid

    def read_scale(self, obj_id, obj_path):
        try:
            scales = pd.read_csv(obj_path + '/scales.csv')
            scales.set_index('obj_id', inplace=True)
            return scales.loc[obj_id].scale
        except Exception as e:
            print 'Error \"{} {}\" while reading scales file. Falling back to scale=1'.format(type(e).__name__, e)
            return 1

    def load_scale(self, fn, pos=None, ori=None, scale=1):
        collision_fn = fn.replace('.obj', '_vhacd.obj')
        if not os.path.exists(collision_fn):
            print('''No collision model for {} was found. Falling back to
                    visual model as collision geometry. This might impact
                    performance of your simulation''').format(fn)
            collision_fn = fn
        visual_fn = fn if self.visual_model else collision_fn

        obj_id = fn.split('/')[-1].replace('.obj', '')

        print scale

        obj = Wavefront(visual_fn)
        size = obj.size
        max_size = size.max()
        #scale = np.clip(max_size, 0.08, 0.9)/max_size * scale
        center = obj.center * scale

        if pos is None:
            pos = np.array([0, 0, max_size * scale])

        if ori is None:
            ori = [0, 0, 0]
        ori = p.getQuaternionFromEuler(ori)

        vid = p.createVisualShape(shapeType=p.GEOM_MESH,fileName=visual_fn,
                rgbaColor=[0.1,0.1,1,1], specularColor=[0.4,.4,0], meshScale=[scale]*3,
                visualFramePosition=-center)
        cid = p.createCollisionShape(shapeType=p.GEOM_MESH,
                fileName=collision_fn, meshScale=[scale]*3,
                collisionFramePosition=-center)
        bid =  p.createMultiBody(baseMass=max_size*scale,baseInertialFramePosition=[0,0,0], baseCollisionShapeIndex=cid, baseVisualShapeIndex = vid, basePosition=pos, baseOrientation=ori)

        if self.debug:
            self.drawFrame([0,0,0], bid)

        return bid

    def transform(self, bid, position=None, rotation=None, scale=None):
        if position is not None:
            _, ori = self._get_pos_and_ori(bid)
            p.resetBasePositionAndOrientation(bid,position, ori)
        if rotation is not None:
            pos,_ = self._get_pos_and_ori(bid)
            rot = p.getQuaternionFromEuler(rotation)
            p.resetBasePositionAndOrientation(bid,pos, rot)

    def replay(self, log_fn, scene_fn, models_path=os.environ['MODELS_PATH']):
        self.restore(scene_fn, models_path)
        if not self.gid:
            self.add_gripper(MODULE_PATH + '/gripper.urdf')
        if not self.bin:
            self.bin = p.loadURDF(MODULE_PATH + '/bin.urdf', basePosition=self.bin_pos)
        log = self._read_logfile(log_fn, verbose=False)

        for idx, record in enumerate(log):
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
            if self.logger is not None:
                self.logger.log()

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
        p.enableJointForceTorqueSensor(self.gid, 2)
        p.enableJointForceTorqueSensor(self.gid, 5)
        p.enableJointForceTorqueSensor(self.gid, 6)
        p.enableJointForceTorqueSensor(self.gid, 7)

    def evaluate_6dof_grasp(self, pose, width):
        if not self.gid:
            self.add_gripper(MODULE_PATH + '/gripper.urdf')
        else:
            for jid in [0,1,3,4,5]:
                p.resetJointState(self.gid, jid, 0)
            p.resetJointState(self.gid, 2, 1)
        if not self.bin:
            self.bin = p.loadURDF(MODULE_PATH + '/bin.urdf', basePosition=self.bin_pos)

        # Pre grasp pose

    def evaluate_grasp(self, pose, width):
        if not self.gid:
            self.add_gripper(MODULE_PATH + '/gripper.urdf')
            #self.add_gripper(MODULE_PATH + '/gripper_wide.urdf')
        else:
            for jid in [0,1,3,4,5]:
                p.resetJointState(self.gid, jid, 0)
            p.resetJointState(self.gid, 2, 1)
        if not self.bin:
            self.bin = p.loadURDF(MODULE_PATH + '/bin.urdf', basePosition=self.bin_pos)

        #Post and pregrasp poses
        pregrasp = pose + [0, 0, .5, 0, 0, 0]
        postgrasp = np.array(p.getBasePositionAndOrientation(self.bin)[0])
        postgrasp = np.concatenate((postgrasp, pose[3:6]))
        postgrasp[2] += 0.75
        bid = self.bodies.next()
        # Move to pregrasp
        self.set_gripper_width(width)
        self.move_gripper_to(pregrasp)
        # Move to grasp
        self.move_gripper_to(pose)
        self.close_gripper()
        # Take offset between object's COM and gripper to compute postgrasp
        gripper_pos = np.array(p.getLinkState(self.gid, 5)[0])
        object_pos = p.getBasePositionAndOrientation(bid)[0]
        offset = gripper_pos[:2] - object_pos[:2]
        postgrasp[:2] += offset
        # Move to postgrasp
        #self.move_gripper_to(pose + np.array([0, 0, 1.5, 0, 0, 0]))
        self.move_gripper_to(postgrasp)
        self.run(epochs=int(0.5/self.timestep)) # Let inertia die at dropoff point. This prevents object shooting out
        self.open_gripper()
        self.run(epochs=int(1./self.timestep)) # Let object fall
        final_pos,_ = p.getBasePositionAndOrientation(bid)
        #print('Final pos: ', final_pos)
        result = np.linalg.norm(final_pos[0:2] - np.array(self.bin_pos[0:2])) < 0.7
        #print('Result: {}'.format(result))
        return result

    def evaluate_3dof_grasp(self, position, approach, orientation, width):
        """
        :param position: position of the grasp
        :param approach: direction the hand will follow along its forward axis to approach the position point
        :param orientation: rotation along the forward axis that the hand will arrive the position at.
        :param width: distance between the plates before arriving to the grasp point
        :return: True/False whether the grasp was successful
        """

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
            pos, ori = self._get_pos_and_ori(k)
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
            csv.write('obj_id,x,y,z,r,p,y,scale\n')
            for bid in self.bodies:
                obj_id = self.obj_ids[bid]
                pos, ori = p.getBasePositionAndOrientation(bid)
                ori = p.getEulerFromQuaternion(ori)
                sc = self.get_object_scale(bid)
                obj_str = ','.join([obj_id] + [str(ps) for ps in pos] + [ str(o) for o in ori] + [str(sc)]) + '\n'
                csv.write(obj_str)

    def get_state(self):
        state = []
        for bid in self.bodies:
            obj_id = self.obj_ids[bid]
            pos, ori = p.getBasePositionAndOrientation(bid)
            ori = p.getEulerFromQuaternion(ori)
            sc = self.get_object_scale(bid)
            obj_str = ','.join([obj_id] + [str(ps) for ps in pos] + [ str(o) for o in ori] + [str(sc)]) + '\n'
            state.append(obj_str)
        return '\n'.join(state)

    def restore(self, csv, path, reset=True):
        """
            Will read the .csv and .bullet files associated to {fn} and load
            the world accordingly. Will look for linked .obj files in the
            {path} directory
        """
        if reset:
            self.reset()
        lines = csv.split('\n')
        for line in lines:
            if line == '' or line[:6] == 'obj_id':
                continue
            fields = line.split(',')
            obj_fn = os.path.join(path, fields[0] + '.obj')
            obj_pos = [float(fields[1]),float(fields[2]),float(fields[3])]
            obj_ori = [float(fields[4]),float(fields[5]),float(fields[6])]
            scale = float(fields[7])
            self.load(obj_fn, obj_pos, obj_ori, scale)
            print('Loaded '+ obj_fn)

        self.cam.target = self.get_clutter_center().tolist()

    def get_clutter_center(self):
        # Find the center of all the objects in the scene
        nbodies = 0
        pos = np.zeros(3)
        for bid in self.bodies:
            nbodies += 1
            pos += np.array(p.getBasePositionAndOrientation(bid)[0])
        pos /= nbodies
        return pos

    def get_object_scale(self, bid):
        return p.getVisualShapeData(bid)[0][3][0]

    def _update_pos(self):
        for id in self.bodies:
            self.old_poses[id] = self._get_pos_and_ori(id)

    def _get_pos_and_ori(self, bId):
        pos, ori = p.getBasePositionAndOrientation(bId)
        return np.array(pos), np.array(ori)

    def drawFrame(self, frame, parent=-1):
        f = np.tile(np.array(frame), (3,1))
        t = f + np.eye(3)

        p.addUserDebugLine(f[0,:], t[0,:], red, parentObjectUniqueId=parent)
        p.addUserDebugLine(f[1,:], t[1,:], green, parentObjectUniqueId=parent)
        p.addUserDebugLine(f[2,:], t[2,:], blue, parentObjectUniqueId=parent)

    def close_gripper(self):
        return self.set_gripper_width(0)

    def open_gripper(self):
        self.set_gripper_width(0.3)

    def set_gripper_width(self, width, vel=0.3, force=1000):
        width = 0.3 - width
        for joint in [6,7]:
            p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                    targetPosition=width/2.,targetVelocity=0, maxVelocity=vel,
                    force=force)
        self.run(epochs=int(0.6/self.timestep))

    def move_gripper_to(self, pose, pos_tol=0.01, ang_tol=2, linvel=0.5, angvel=2, force=1000):
        pose[2] += 0.005
        max_dist = 2. # Max distance that the gripper will have to traverse. This determines the timeout
        timeout = max_dist/linvel
        try:
            ang_tol = np.radians(ang_tol)
            # Linear joints
            for joint in range(3):
                p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                        targetPosition=pose[joint], targetVelocity=0,
                        force=force, maxVelocity=linvel, positionGain=0.3, velocityGain=1)
            # Rotational joints
            for joint in [3, 4, 5]:
                p.setJointMotorControl2(self.gid, joint, p.POSITION_CONTROL,
                        targetPosition=pose[joint], targetVelocity=0, force=force,
                        maxVelocity=angvel, positionGain=0.3, velocityGain=1)
            for _ in range(int(timeout/self.timestep)):
                state = p.getLinkState(self.gid, 5)
                pos = np.array(state[0])
                ori = np.array(p.getEulerFromQuaternion(state[1]))
                ori = np.array([x[0] for x in p.getJointStates(self.gid, [3,4,5])])
                #print('Offset {} {}'.format(pose[0:3] - pos,
                #    pose[3:6] - ori))
                if np.linalg.norm(pos - pose[0:3]) < pos_tol and (np.abs(ori - pose[3:6]) < ang_tol).all():
                    # print('Arrived within tolerance')
                    break
                self.step()
            else:
                # print('Move timed out')
                pass
        except KeyboardInterrupt:
            print 'Cancel move'
            return
        if self.debug:
            self.cam.snap()

    def teleport_to_pose(self, position, orientation, width):
        """
        Instantaneously moves the gripper to the given configuration without making use of the physics engine. Useful
        for quickly reaching pre-grasps
        :param position: vec3
        :param orientation: vec3 intrinsic eulers XYZ
        :param width: float
        :return:
        """
        p.resetJointState(self.gid, 0, position[0])
        p.resetJointState(self.gid, 1, position[1])
        p.resetJointState(self.gid, 2, position[2])

        p.resetJointState(self.gid, 3, orientation[0])
        p.resetJointState(self.gid, 4, orientation[1])
        p.resetJointState(self.gid, 5, orientation[2])

        width = 0.3 - width
        width /= 2

        p.resetJointState(self.gid, 6, width)
        p.resetJointState(self.gid, 7, width)

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
            self.step()
            if autostop and self.is_stable():
                print('Stable')
                break
            self._update_pos()
        else:
            if autostop:
                print('Unstable')

    def run_debug_teleport(self):
        states = map(lambda joint: joint[0], p.getJointStates(self.gid, range(6)))

        sliders = [
            p.addUserDebugParameter('x', -1, 1, states[0]),
            p.addUserDebugParameter('y', -1, 1, states[1]),
            p.addUserDebugParameter('z', -1, 1, states[2]),
            p.addUserDebugParameter('roll', -180, 180, np.rad2deg(states[3])),
            p.addUserDebugParameter('pitch', -180, 180, np.rad2deg(states[4])),
            p.addUserDebugParameter('yaw', -180, 180, np.rad2deg(states[5])),
            p.addUserDebugParameter('width', 0, 0.3, 0.3)
        ]

        while True:
            targets_values = [p.readUserDebugParameter(s_id) for s_id in sliders]
            self.teleport_to_pose(targets_values[:3], np.deg2rad(targets_values[3:6]), targets_values[6])

    def run_generate_scene(self):
        body_id = self.bodies.next()

        pos, ori = self._get_pos_and_ori(body_id)
        eulers = p.getEulerFromQuaternion(ori)
        sliders = [
            p.addUserDebugParameter('x', -2, 2, pos[0]),
            p.addUserDebugParameter('y', -2, 2, pos[1]),
            p.addUserDebugParameter('z', -2, 2, pos[2]),
            p.addUserDebugParameter('roll', -180, 180, np.rad2deg(eulers[0])),
            p.addUserDebugParameter('pitch', -180, 180, np.rad2deg(eulers[1])),
            p.addUserDebugParameter('yaw', -180, 180, np.rad2deg(eulers[2]))
        ]

        qKey = ord('q')
        pKey = ord(' ')
        nextKey = 65309L

        simulate = False

        stop = False

        while True:
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                stop = True
                break
            if nextKey in keys and keys[nextKey] & p.KEY_WAS_TRIGGERED:
                break
            if pKey in keys and keys[pKey] & p.KEY_WAS_TRIGGERED:
                pos, ori = self._get_pos_and_ori(body_id)
                simulate = not simulate

            self.step()
            if not simulate:
                targets_values = [p.readUserDebugParameter(s_id) for s_id in sliders]
                p.resetBasePositionAndOrientation(body_id, targets_values[:3], p.getQuaternionFromEuler(np.deg2rad(targets_values[3:6])))

        p.removeAllUserDebugItems()
        for slider in sliders:
            p.removeUserDebugItem(slider)
        return stop

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
            self.step()
            values = [p.readUserDebugParameter(id) for id in range(7)]
            values[6] /= 2.
            values.append(-values[6])
            p.setJointMotorControlArray(bodyUniqueId=self.gid, jointIndices=range(8),
                    controlMode=p.POSITION_CONTROL, targetPositions=values)

    def run_action(self, tolerance=1e-2):
        self.step()
        velocities = lambda: [joint[1] for joint in p.getJointStates(self.gid, range(8))]

        for _ in range(int(self.timeout/self.timestep)):
            if (np.abs(np.array(velocities())) < tolerance).all():
                break
            self.step()
        else:
            print('Action timed out')

    def step(self):
        p.stepSimulation()
        self.sleep()
        if self.logger is not None:
            self.logger.log()

    def start_log(self, fn, logger=VIDEO_LOGGER, rate=25):
        assert logger == VIDEO_LOGGER or logger == STATE_LOGGER or logger == OPENGL_LOGGER
        if logger == VIDEO_LOGGER:
            self.logger = VideoLogger(fn, self.timestep, rate)
        elif logger == STATE_LOGGER:
            self.logger = StateLogger(fn)
        elif logger == OPENGL_LOGGER:
            self.logger = OpenGLLogger(fn)

    def stop_log(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None

    def disconnect(self):
        p.disconnect()

    def add_debug_pose(self, position, z, x, width):
        position = np.array(position)
        z = np.array(z)
        z = z / np.linalg.norm(z)
        x = np.array(x)
        x = x / np.linalg.norm(x) * width / 2.
        p.addUserDebugLine(position, position + z, [0, 0, 1])
        p.addUserDebugLine(position, position + x, [1, 0, 0])
        p.addUserDebugLine(position, position - x, [1, 0, 0])

