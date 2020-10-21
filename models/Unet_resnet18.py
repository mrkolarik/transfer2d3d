# -------------------------------------  IMPORTS ----------------------------------
import numpy as np
import tensorflow as tf
# from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, add, concatenate, BatchNormalization, \
    Activation, Dropout
from keras import backend, layers

np.random.seed(1337)
tf.logging.set_verbosity(tf.logging.FATAL)
tf.set_random_seed(1337)

backend.set_image_data_format('channels_last')

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(train_encoder, **params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
        'trainable': train_encoder
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(train_encoder, **params):
    axis = -1 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
        'trainable': train_encoder
    }
    default_bn_params.update(params)
    return default_bn_params

# -------------------------------------  MODEL ----------------------------------

def res_unet_resnet18(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):

    # Block 1
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(train_encoder, scale=False)
    bn_params = get_bn_params(train_encoder)
    conv_params = get_conv_params(train_encoder)
    init_filters = 64

    inputs = Input((image_depth, image_rows, image_cols, input_channels))
    # Block 1
    # get parameters for model layers
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(inputs)
    x = layers.ZeroPadding3D(padding=(0, 3, 3))(x)
    x = layers.Conv3D(init_filters, (1, 7, 7), strides=(1, 1, 1), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    skip_connection_1 = x
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.MaxPooling3D((2, 3, 3), strides=(2, 2, 2), padding='valid', name='pooling0')(x)

    # Stage 1, Unit 1 - Settings
    stage = 0
    block = 0
    strides = (1, 1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    # defining shortcut connection
    shortcut = layers.Conv3D(filters, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 1, Unit 2 - Settings
    stage = 0
    block = 1
    strides = (1, 1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 2, Unit 1 - Settings
    stage = 1
    block = 0
    strides = (2, 2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_2 = x
    # defining shortcut connection
    shortcut = layers.Conv3D(filters, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (2, 3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 2, Unit 2 - Settings
    stage = 1
    block = 1
    strides = (1, 1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 3, Unit 1 - Settings
    stage = 2
    block = 0
    strides = (2, 2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_3 = x
    # defining shortcut connection
    shortcut = layers.Conv3D(filters, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (2, 3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 3, Unit 2 - Settings
    stage = 2
    block = 1
    strides = (1, 1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 4, Unit 1 - Settings
    stage = 3
    block = 0
    strides = (2, 2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_4 = x
    # defining shortcut connection
    shortcut = layers.Conv3D(filters, (1, 1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (2, 3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 4, Unit 2 - Settings
    stage = 3
    block = 1
    strides = (1, 1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding3D(padding=(0, 1, 1))(x)
    x = layers.Conv3D(filters, (1, 3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Resnet OUTPUT
    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)

    conv51 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv51', trainable=train_encoder)(x)
    conv52 = Conv3D(512, (1, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv52', trainable=train_encoder)(conv51)
    conc52 = concatenate([x, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans6')(conc52), skip_connection_4], axis=4)
    conv61 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv61')(up6)
    conv62 = Conv3D(256, (2, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv62')(conv61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans7')(conc62), skip_connection_3], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv71')(up7)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv72')(conv71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans8')(conc72), skip_connection_2], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv81')(up8)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv82')(conv81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same', name='trans9')(conc82), skip_connection_1], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 2, 2), padding='same', name='conv91')(up9)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', dilation_rate=(1, 1, 1), padding='same', name='conv92')(conv91)
    conc92 = concatenate([up9, conv92], axis=4)


    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='conv10')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model