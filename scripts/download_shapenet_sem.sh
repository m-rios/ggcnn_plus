#! /bin/bash
curl --request GET -L -O -C - \
     --url 'http://shapenet.cs.stanford.edu/shapenet/obj-zip/ShapeNetSem.v0/models-OBJ.zip'\
     --output "${THESIS_DATA_PATH}/3d_models/shapenetsem"