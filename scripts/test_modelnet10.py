import os
import glob
import datetime
import numpy as np
import pylab as plt
import core.network as network
from core.network import Network
from skimage import io
from skimage.transform import resize

dt = datetime.datetime.now().strftime('%y%m%d_%H%M')

# Paths for MBA
# DS_PATH = '/Users/mario/Developer/msc-thesis/data/datasets/raw/ortographic_modelnet10_dataset_' \
#           '/ortographic_modelnet10_dataset_gray_images'
# RESULTS_PATH = os.path.join(os.environ['RESULTS_PATH'], 'modelnet10_{}'.format(dt))
# MODEL_FNS = ['/Users/mario/Developer/msc-thesis/data/networks/ggcnn_rss/epoch_29_model.hdf5',
#              '/Users/mario/Developer/msc-thesis/data/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5']

# Paths for Peregrine
DS_PATH = '/home/s3485781/DATA/datasets/raw/ortographic_modelnet10_dataset_' \
          '/ortographic_modelnet10_dataset_gray_images'
RESULTS_PATH = os.path.join(os.environ['RESULTS_PATH'], 'modelnet10_{}'.format(dt))
MODEL_FNS = ['/home/s3485781/DATA/networks/ggcnn_rss/epoch_29_model.hdf5',
             '/home/s3485781/DATA/networks/beam_search_2/depth_3_arch_9x9x32_5x5x16_3x3x16_3x3x16_3x3x8_3x3x8_epoch_3_model.hdf5']

image_fns = glob.glob(os.path.join(DS_PATH, '*/*/*.jpg'))
image_ids = ['/'.join(fn.split('/')[-3:]) for fn in image_fns]

# images = np.zeros((len(image_fns), 300, 300))
images = np.zeros((10, 300, 300))

for idx, image_fn in enumerate(image_fns):
    # Upscale images to 300x300
    images[idx] = resize(io.imread(image_fn, as_gray=True), (300, 300), anti_aliasing=True, mode='constant')
    # Reverse values to denote distance from camera
    images[idx] = np.max(images[idx]) - images[idx]
    if idx == 9:
        break

for model_fn in MODEL_FNS:
    model_name = model_fn.split('/')[-2]
    rss = Network(model_fn)
    pos, ang, wid = rss.predict(images)

    for idx in range(images.shape[0]):
        output_fn = os.path.join(RESULTS_PATH, model_name, image_ids[idx])
        output_path = '/'.join(output_fn.split('/')[:-1])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        network.save_output_plot(images[idx], pos[idx], ang[idx], wid[idx], filename=output_fn)
