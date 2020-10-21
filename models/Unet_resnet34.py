# -------------------------------------  IMPORTS ----------------------------------
import numpy as np
import tensorflow as tf
# from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, add, concatenate, BatchNormalization, \
    Activation, Dropout
from keras import backend as K

np.random.seed(1337)
tf.logging.set_verbosity(tf.logging.FATAL)
tf.set_random_seed(1337)

K.set_image_data_format('channels_last')

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"





# -------------------------------------  MODEL ----------------------------------


def dense_unet(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 1, train_encoder = True):
    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block1_conv1', trainable=train_encoder)(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block1_conv2', trainable=train_encoder)(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block2_conv1', trainable=train_encoder)(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block2_conv2', trainable=train_encoder)(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv1', trainable=train_encoder)(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block3_conv2', trainable=train_encoder)(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv3', trainable=train_encoder)(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv1', trainable=train_encoder)(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    pool4 = MaxPooling3D(pool_size=(1, 2, 2))(conc42)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv2', trainable=train_encoder)(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv3', trainable=train_encoder)(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(1, 2, 2), padding='same', name='trans6')(conc52), conc42], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(1, 2, 2), padding='same', name='trans7')(conc62), conc32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), conc22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), conc12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def res_unet(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 1, train_encoder = True):
    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block1_conv1', trainable=train_encoder)(inputs)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block1_conv2', trainable=train_encoder)(conv11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block2_conv1', trainable=train_encoder)(pool1)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block2_conv2', trainable=train_encoder)(conv21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv1', trainable=train_encoder)(pool2)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block3_conv2', trainable=train_encoder)(conv31)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv3', trainable=train_encoder)(pool3)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv1', trainable=train_encoder)(conv41)
    conc42 = concatenate([pool3, conv42], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv2', trainable=train_encoder)(pool4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv3', trainable=train_encoder)(conv51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(conc52), conc42], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conv61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(conc62), conc32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conv71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), conc22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conv81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), conc12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conv91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def res_unet_batchnorm(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 1, train_encoder = True):
    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block1_conv1', trainable=train_encoder)(inputs)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block1_conv2', trainable=train_encoder)(conv11)
    conc12 = concatenate([inputs, conv12], axis=4)
    batch1 = BatchNormalization(name='batchnorm_1')(conc12)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(batch1)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block2_conv1', trainable=train_encoder)(pool1)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block2_conv2', trainable=train_encoder)(conv21)
    conc22 = concatenate([pool1, conv22], axis=4)
    batch2 = BatchNormalization(name='batchnorm_2')(conc22)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(batch2)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv1', trainable=train_encoder)(pool2)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block3_conv2', trainable=train_encoder)(conv31)
    conc32 = concatenate([pool2, conv32], axis=4)
    batch3 = BatchNormalization(name='batchnorm_3')(conc32)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(batch3)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='block3_conv3', trainable=train_encoder)(pool3)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv1', trainable=train_encoder)(conv41)
    conc42 = concatenate([pool3, conv42], axis=4)
    batch4 = BatchNormalization(name='batchnorm_4')(conc42)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(batch4)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv2', trainable=train_encoder)(pool4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='block4_conv3', trainable=train_encoder)(conv51)
    conc52 = concatenate([pool4, conv52], axis=4)
    batch5 = BatchNormalization(name='batchnorm_5')(conc52)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(batch5), conc42], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conv61)
    conc62 = concatenate([up6, conv62], axis=4)
    batch6 = BatchNormalization(name='batchnorm_6')(conc62)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(batch6), conc32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conv71)
    conc72 = concatenate([up7, conv72], axis=4)
    batch7 = BatchNormalization(name='batchnorm_7')(conc72)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(batch7), conc22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conv81)
    conc82 = concatenate([up8, conv82], axis=4)
    batch8 = BatchNormalization(name='batchnorm_8')(conc82)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(batch8), conc12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conv91)
    conc92 = concatenate([up9, conv92], axis=4)
    batch9 = BatchNormalization(name='batchnorm_9')(conc92)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(batch9)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def res_unet_vgg(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    block1_conv1 = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train_encoder)(inputs)
    block1_conv2 = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train_encoder)(block1_conv1)
    block1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    block2_conv1 = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train_encoder)(block1_pool)
    block2_conv2 = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train_encoder)(block2_conv1)
    block2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    block3_conv1 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train_encoder)(block2_pool)
    block3_conv2 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train_encoder)(block3_conv1)
    block3_conv3 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train_encoder)(block3_conv2)
    block3_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(block3_conv3)

    # Block 4
    block4_conv1 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train_encoder)(block3_pool)
    block4_conv2 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train_encoder)(block4_conv1)
    block4_conv3 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train_encoder)(block4_conv2)
    block4_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(block4_conv3)

    # Block 5
    # block5_conv1 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train_encoder)(block4_pool)
    # block5_conv2 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train_encoder)(block5_conv1)
    # block5_conv3 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train_encoder)(block5_conv2)

    conv51 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv51', trainable=train_encoder)(block4_pool)
    conv52 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv52', trainable=train_encoder)(conv51)
    conc52 = concatenate([block4_pool, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(conc52), block4_conv3], axis=4)
    conv61 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conv62 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conv61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(conc62), block3_conv3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conv71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), block2_conv2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conv81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), block1_conv2], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conv91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def res_unet_resnet18(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    block1_conv1 = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv1', trainable=train_encoder)(inputs)
    block1_conv2 = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv2', trainable=train_encoder)(block1_conv1)
    block1_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    block2_conv1 = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv1', trainable=train_encoder)(block1_pool)
    block2_conv2 = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv2', trainable=train_encoder)(block2_conv1)
    block2_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    block3_conv1 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv1', trainable=train_encoder)(block2_pool)
    block3_conv2 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv2', trainable=train_encoder)(block3_conv1)
    block3_conv3 = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv3', trainable=train_encoder)(block3_conv2)
    block3_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(block3_conv3)

    # Block 4
    block4_conv1 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv1', trainable=train_encoder)(block3_pool)
    block4_conv2 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv2', trainable=train_encoder)(block4_conv1)
    block4_conv3 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv3', trainable=train_encoder)(block4_conv2)
    block4_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(block4_conv3)

    # Block 5
    # block5_conv1 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train_encoder)(block4_pool)
    # block5_conv2 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train_encoder)(block5_conv1)
    # block5_conv3 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train_encoder)(block5_conv2)

    conv51 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv51', trainable=train_encoder)(block4_pool)
    conv52 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv52', trainable=train_encoder)(conv51)
    conc52 = concatenate([block4_pool, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(conc52), block4_conv3], axis=4)
    conv61 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conv62 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conv61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(conc62), block3_conv3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conv71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), block2_conv2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conv81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), block1_conv2], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conv91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model