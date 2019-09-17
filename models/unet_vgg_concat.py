from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
tf.set_random_seed(256)

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling3D, ZeroPadding3D, Add
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
from matplotlib import pyplot


def get_unet():
    inputs = Input((400, 400, 1))
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv22 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

    conv31 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv32 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv31)
    conv33 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)

    conv41 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv42 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv41)
    conv43 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(conv42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv43)

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv51)
    conv53 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv52)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv53), conv43], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv33], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv22], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv12], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    # model.summary()
    # #plot_model(model, to_file='model.png')
    #
    # model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

    return model