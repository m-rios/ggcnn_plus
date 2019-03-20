#!/usr/bin/env python

import os
from datetime import datetime
import argparse

import h5py
import numpy as np
import multiprocessing

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Input
from keras.models import Model
from keras.utils import Sequence

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Path to dataset')
args = parser.parse_args()

INPUT_DATASET = args.dataset
BATCH_SZ = 24
N_EPOCHS = 100
WORKERS = multiprocessing.cpu_count()

FILTER_SIZES = [
    [(9, 9), (5, 5), (3, 3)]
]

NO_FILTERS = [
    [32, 16, 8],
]

BATCH_SZ = 4

class JacquardGenerator(Sequence):
    def __init__(self, path, batch_sz, dataset='train'):
        self.ds = h5py.File(path, 'r')[dataset]
        self.n_samples = self.ds['img_id'].size
        self.batch_sz = batch_sz

    def __len__(self):
        return np.ceil(self.n_samples/float(self.batch_sz))

    def __getitem__(self, idx):

        fr = idx*self.batch_sz
        to = (idx+1)*self.batch_sz

        depth = np.expand_dims(np.array(self.ds['depth_inpainted'][fr:to]), -1)
        point = np.expand_dims(np.array(self.ds['grasp_points_img'][fr:to]), -1)
        angle = np.array(self.ds['angle_img'][fr:to])
        cos = np.expand_dims(np.cos(2*angle), -1)
        sin = np.expand_dims(np.sin(2*angle), -1)
        grasp_width = np.expand_dims(np.array(self.ds['grasp_width'][fr:to]), -1)
        grasp_width = np.clip(grasp_width, 0, 150)/150.0

        batch_x = depth
        batch_y = [point, cos, sin, grasp_width]

        return batch_x, batch_y


train_generator = JacquardGenerator(INPUT_DATASET, BATCH_SZ, dataset='train')
x_train = train_generator[0][0]
test_generator = JacquardGenerator(INPUT_DATASET, BATCH_SZ, dataset='test')


for filter_sizes in FILTER_SIZES:
    for no_filters in NO_FILTERS:
        dt = datetime.now().strftime('%y%m%d_%H%M')

        NETWORK_NAME = "ggcnn_%s_%s_%s__%s_%s_%s" % (filter_sizes[0][0], filter_sizes[1][0], filter_sizes[2][0],
                                                     no_filters[0], no_filters[1], no_filters[2])
        NETWORK_NOTES = """
            Input: Inpainted depth, subtracted mean, in meters, with random rotations and zoom. 
            Output: q, cos(2theta), sin(2theta), grasp_width in pixels/150.
            Dataset: %s
            Filter Sizes: %s
            No Filters: %s
        """ % (
            INPUT_DATASET,
            repr(filter_sizes),
            repr(no_filters)
        )
        OUTPUT_FOLDER = 'data/networks/%s__%s/' % (dt, NETWORK_NAME)

        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        # Save the validation data so that it matches this network.
        # np.save(os.path.join(OUTPUT_FOLDER, '_val_input'), x_test)

        # ====================================================================================================
        # Network

        input_layer = Input(shape=x_train.shape[1:])

        x = Conv2D(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu')(input_layer)
        x = Conv2D(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu')(x)
        encoded = Conv2D(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu')(x)

        x = Conv2DTranspose(no_filters[2], kernel_size=filter_sizes[2], strides=(2, 2), padding='same', activation='relu')(encoded)
        x = Conv2DTranspose(no_filters[1], kernel_size=filter_sizes[1], strides=(2, 2), padding='same', activation='relu')(x)
        x = Conv2DTranspose(no_filters[0], kernel_size=filter_sizes[0], strides=(3, 3), padding='same', activation='relu')(x)

        # ===================================================================================================
        # Output layers

        pos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='pos_out')(x)
        cos_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='cos_out')(x)
        sin_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='sin_out')(x)
        width_output = Conv2D(1, kernel_size=2, padding='same', activation='linear', name='width_out')(x)

        # ===================================================================================================
        # And go!

        ae = Model(input_layer, [pos_output, cos_output, sin_output, width_output])
        ae.compile(optimizer='rmsprop', loss='mean_squared_error')

        ae.summary()

        with open(os.path.join(OUTPUT_FOLDER, '_description.txt'), 'w') as f:
            # Write description to file.
            f.write(NETWORK_NOTES)
            f.write('\n\n')
            ae.summary(print_fn=lambda q: f.write(q + '\n'))

        with open(os.path.join(OUTPUT_FOLDER, '_dataset.txt'), 'w') as f:
            # Write dataset name to file for future reference.
            f.write(INPUT_DATASET)

        tb_logdir = './data/tensorboard/%s_%s' % (dt, NETWORK_NAME)

        my_callbacks = [
            TensorBoard(log_dir=tb_logdir),
            ModelCheckpoint(os.path.join(OUTPUT_FOLDER, 'epoch_{epoch:02d}_model.hdf5'), period=1),
        ]

        #ae.fit_generator(generator=train_generator,
        #        steps_per_epoch=train_generator.n_samples // BATCH_SZ,
        #        epochs=N_EPOCHS,
        #        verbose=1,
        #        validation_data=test_generator,
        #        validation_steps=test_generator.n_samples // BATCH_SZ,
        #        use_multiprocessing=True,
        #        workers=WORKERS,
        #        max_queue_size=32,
        #        shuffle=True,
        #        callbacks=my_callbacks)
        ae.fit_generator(generator=train_generator,
                steps_per_epoch=train_generator.n_samples // BATCH_SZ,
                epochs=N_EPOCHS,
                verbose=1,
                validation_data=test_generator,
                validation_steps=test_generator.n_samples // BATCH_SZ,
                shuffle=True,
                callbacks=my_callbacks)
