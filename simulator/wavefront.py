import numpy as np

class Obj(object):

    def __init__(self, fn):
        self.vertices = np.array([], (None, 3))
        with open(fn, 'r') as f:
            for line in f.readlines():
                fields = line.split(' ')
                if fields[0] == 'v':
                    self.vertices = np.append(self.vertices, [[float(x) for x in fields[1:4]]], axis=0)

    @property
    def aabb(self):
        lower = np.min(self.vertices, axis=0)
        upper = np.max(self.vertices, axis=0)
        return np.vstack((lower, upper))

    @property
    def center(self):
        return np.mean(self.vertices, axis=0)

    @property
    def size(self):
        bb = self.aabb
        return np.abs(bb[1] - bb[0])

if __name__ == '__main__':
    obj_fn = '/mnt/d/models/3f91158956ad7db0322747720d7d37e8.obj'

    obj = Obj(obj_fn)
    print 'Center', obj.center
    print 'AABB', obj.aabb
    print 'Size', obj.size
