import random
import os
import argparse

from zipfile import ZipFile

class Getch:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

parser = argparse.ArgumentParser()
parser.add_argument('shapenet_path', help='directory where shapenet .obj files will be looked for')
parser.add_argument('output_path', help='directory where selected models will be copied to')
args = parser.parse_args()

if not os.path.exists(os.path.join(args.output_path, 'models')):
    os.makedirs(os.path.join(args.output_path, 'models'))

getch = Getch()
selected = 0
with ZipFile(args.shapenet_path, 'r') as z:
    object_fns = filter(lambda x: '.obj' in x, z.namelist())
    random.shuffle(object_fns)
    for object_fn in object_fns:
        z.extract(object_fn)
        os.system('qlmanage -p {}'.format(object_fn))
        answer = raw_input('Type y/n \n to accept/ignore')
        if answer == 'y':
            os.rename(object_fn, os.path.join(args.output_path, object_fn))
            selected += 1
            print 'You have selected {} already'.format(selected)
        else:
            print 'Skipping'
            os.remove(object_fn)



