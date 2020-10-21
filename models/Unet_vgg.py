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


def res_unet_vgg_batchnorm(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
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

    conv51 = Conv3D(512, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv51', trainable=train_encoder)(block4_pool)
    drop51 = Dropout(0.15)(conv51)
    conv52 = Conv3D(512, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv52', trainable=train_encoder)(drop51)
    drop52 = Dropout(0.15)(conv52)
    batch5 = BatchNormalization(name='batchnorm_5')(drop52)
    conc52 = concatenate([block4_pool, batch5], axis=4)


    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans6')(conc52), block4_conv3], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv61')(up6)
    drop61 = Dropout(0.15)(conv61)
    conv62 = Conv3D(256, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv62')(drop61)
    drop62 = Dropout(0.15)(conv62)
    batch6 = BatchNormalization(name='batchnorm_6')(drop62)
    conc62 = concatenate([up6, batch6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans7')(conc62), block3_conv3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv71')(up7)
    drop71 = Dropout(0.15)(conv71)
    conv72 = Conv3D(128, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv72')(drop71)
    drop72 = Dropout(0.15)(conv72)
    batch7 = BatchNormalization(name='batchnorm_7')(drop72)
    conc72 = concatenate([up7, batch7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans8')(conc72), block2_conv2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv81')(up8)
    drop81 = Dropout(0.15)(conv81)
    conv82 = Conv3D(64, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv82')(drop81)
    drop82 = Dropout(0.15)(conv82)
    batch8 = BatchNormalization(name='batchnorm_8')(drop82)
    conc82 = concatenate([up8, batch8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans9')(conc82), block1_conv2], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv91')(up9)
    drop91 = Dropout(0.15)(conv91)
    conv92 = Conv3D(32, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv92')(drop91)
    drop92 = Dropout(0.15)(conv92)
    batch9 = BatchNormalization(name='batchnorm_9')(drop92)
    conc92 = concatenate([up9, batch9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def res_unet_vgg_dropout(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
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

    conv51 = Conv3D(512, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv51', trainable=train_encoder)(block4_pool)
    drop51 = Dropout(0.15)(conv51)
    conv52 = Conv3D(512, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv52', trainable=train_encoder)(drop51)
    drop52 = Dropout(0.15)(conv52)
    conc52 = concatenate([block4_pool, drop52], axis=4)


    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans6')(conc52), block4_conv3], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv61')(up6)
    drop61 = Dropout(0.15)(conv61)
    conv62 = Conv3D(256, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv62')(drop61)
    drop62 = Dropout(0.15)(conv62)
    conc62 = concatenate([up6, drop62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans7')(conc62), block3_conv3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv71')(up7)
    drop71 = Dropout(0.15)(conv71)
    conv72 = Conv3D(128, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv72')(drop71)
    drop72 = Dropout(0.15)(conv72)
    conc72 = concatenate([up7, drop72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans8')(conc72), block2_conv2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv81')(up8)
    drop81 = Dropout(0.15)(conv81)
    conv82 = Conv3D(64, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv82')(drop81)
    drop82 = Dropout(0.15)(conv82)
    conc82 = concatenate([up8, drop82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='btrans9')(conc82), block1_conv2], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='elu', dilation_rate=(1, 2, 2), padding='same', name='bconv91')(up9)
    drop91 = Dropout(0.15)(conv91)
    conv92 = Conv3D(32, (3, 3, 3), activation='elu', dilation_rate=(1, 1, 1), padding='same', name='bconv92')(drop91)
    drop92 = Dropout(0.15)(conv92)
    conc92 = concatenate([up9, drop92], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def dense_unet_vgg(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
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

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv51', trainable=train_encoder)(block4_pool)
    conc51 = concatenate([block4_pool, conv51], axis=4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv52', trainable=train_encoder)(conc51)
    conc52 = concatenate([block4_pool, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(conc52), block4_conv3], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(conc62), block3_conv3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), block2_conv2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), block1_conv2], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)


    model = Model(inputs=[inputs], outputs=[conv10])

    return model


def planar_3D_vgg(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
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
    block5_conv1 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv1', trainable=train_encoder)(block4_pool)
    block5_conv2 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv2', trainable=train_encoder)(block5_conv1)
    block5_conv3 = Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv3', trainable=train_encoder)(block5_conv2)

    model = Model(inputs=[inputs], outputs=[block5_conv3])

    return model