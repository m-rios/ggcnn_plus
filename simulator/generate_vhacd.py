from subprocess import Popen
import argparse
import glob
import os

VHACD_PATH = '/home/s3485781/v-hacd/build/linux2/test/testVHACD'
LOG_FILE = '/tmp/vhacd.log'

class VHACD:
    def __init__(self):

        self.vhacd_path = VHACD_PATH
        # Maximum number of voxels generated during the voxelization stage
        self.resolution = 100000
        # Maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane
        self.depth = 20
        # Maximum concavity
        self.concavity = 0.0025
        self.concavity = 0.0
        # Granularity of the search for the "best" clipping plane
        self.planeDownsampling = 4
        # Precision of the convex-hull generation process during the clipping plane selection stage
        self.convexhullDownsampling = 4
        # Bias toward clipping along symmetry planes
        self.alpha = 0.05
        # Bias toward clipping along revolution axes'
        self.beta = 0.05
        # Maximum allowed concavity during the merge stage
        self.gamma = 0.05
        self.gamma = 0.0
        # Enable/disable normalizing the mesh before applying the convex decomposition
        self.pca = False
        # Approximate convex decomposition mode
        self.mode = 'VOXEL'
        # Maximum number of vertices per convex-hull
        self.maxNumVerticesPerCH = 32
        # Minimum volume to add vertices to convex-hulls
        self.minVolumePerCH = 0.0001


    def run(self, in_path, out_path):

        cmd_line = '"{}" --input "{}" --resolution {} --depth {} --concavity {:g} --planeDownsampling {} --convexhullDownsampling {} --alpha {:g} --beta {:g} --gamma {:g} --pca {:b} --mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} --output "{}" --log "{}"'.format(
            self.vhacd_path,
            in_path,
            self.resolution,
            self.depth,
            self.concavity,
            self.planeDownsampling,
            self.convexhullDownsampling,
            self.alpha,
            self.beta,
            self.gamma,
            self.pca,
            self.mode == 'TETRAHEDRON',
            self.maxNumVerticesPerCH,
            self.minVolumePerCH,
            out_path,
            LOG_FILE)

        vhacd_process = Popen(cmd_line, bufsize=-1, close_fds=True, shell=True)
        vhacd_process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to folder with .obj')
    parser.add_argument('output', help='Path to destination')

    args = parser.parse_args()

    vhacd = VHACD()

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    print glob.glob(os.path.join(args.input, '*.obj'))
    for obj_fn in glob.glob(os.path.join(args.input, '*.obj')):
        print 'Processing ' + obj_fn
        obj_name = obj_fn.split('/')[-1].split('.')[0]
        out_fn = os.path.join(args.output, obj_name + '_vhacd.obj')
        vhacd.run(obj_fn, out_fn)

## Maximum number of voxels generated during the voxelization stage
#self.resolution = 100000
## Maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane
#self.depth = 20
## Maximum concavity
#self.concavity = 0.0025
## Granularity of the search for the "best" clipping plane
#self.planeDownsampling = 4
## Precision of the convex-hull generation process during the clipping plane selection stage
#self.convexhullDownsampling = 4
## Bias toward clipping along symmetry planes
#self.alpha = 0.05
## Bias toward clipping along revolution axes'
#self.beta = 0.05
## Maximum allowed concavity during the merge stage
#self.gamma = 0.05
## Enable/disable normalizing the mesh before applying the convex decomposition
#self.pca = False
## Approximate convex decomposition mode
#self.mode = 'VOXEL'
## Maximum number of vertices per convex-hull
#self.maxNumVerticesPerCH = 32
## Minimum volume to add vertices to convex-hulls
#self.minVolumePerCH = 0.0001
