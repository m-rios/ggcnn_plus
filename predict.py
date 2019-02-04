#!/usr/bin/env python
import os
from skimage import io
from matplotlib import pyplot as plt
import argparse
import sys
from keras.models import load_model
from dataset_processing.image import Image, DepthImage
from dataset_processing import grasp
from dataset_processing.grasp import detect_grasps
import numpy as np

OUTPUT_IMG_SIZE = (300, 300)
DATA_PATH = 'data/cornell'

def img_preprocess(img_n):
    pcd_fn = os.path.join(DATA_PATH, 'pcd{}.txt'.format(img_n))
    pos_grasp_fn = os.path.join(DATA_PATH, 'pcd{}cpos.txt'.format(img_n))
    rgb_fn = os.path.join(DATA_PATH, 'pcd{}r.png'.format(img_n))

    # This returns an Image object
    img_depth = DepthImage.from_pcd(pcd_fn, (480, 640))
    img_depth.inpaint()
    img_rgb = Image(io.imread(rgb_fn))

    bounding_boxes_base = grasp.BoundingBoxes.load_from_file(pos_grasp_fn)
    center = bounding_boxes_base.center

    left = max(0, min(center[1] - OUTPUT_IMG_SIZE[1] // 2, img_depth.shape[1] - OUTPUT_IMG_SIZE[1]))
    right = min(img_depth.shape[1], left + OUTPUT_IMG_SIZE[1])

    top = max(0, min(center[0] - OUTPUT_IMG_SIZE[0] // 2, img_depth.shape[0] - OUTPUT_IMG_SIZE[0]))
    bottom = min(img_depth.shape[0], top + OUTPUT_IMG_SIZE[0])

    img_depth.crop((top, left), (bottom, right))
    img_rgb.crop((top, left), (bottom, right))

    img_depth.normalise()

    return img_depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('img_n')
    arguments = parser.parse_args()
    model_fn = arguments.model_file
    img_n = arguments.img_n

    model = load_model(model_fn)

    input_img = img_preprocess(img_n)
    # input_img.show()
    input_data = input_img.img[np.newaxis,:]
    input_data = np.expand_dims(input_data, -1)
    qual, cos, sin, w = model.predict(input_data, batch_size=1)
    angles = np.arctan2(sin, cos)/2.0
    widths = w * 150.0

    grasps = detect_grasps(qual, angles, width_img=widths, no_grasps=1) 
    print(grasps)




