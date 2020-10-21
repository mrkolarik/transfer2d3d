"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers, Model, utils, backend
from keras import backend as K
from classification_models.keras import Classifiers
from keras.utils import plot_model


K.set_image_data_format('channels_last')

import os

WEIGHTS_PATH = ('https://github.com/qubvel/classification_models/'
                'releases/download/0.0.1/'
                'resnet18_imagenet_1000.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/qubvel/classification_models/'
                'releases/download/0.0.1/'
                'resnet18_imagenet_1000_no_top.h5')


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

# -------------------------------------------------------------------------
#   Models
# -------------------------------------------------------------------------


def resnet18_2d_qubvel():
    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model = ResNet18((256, 256, 3), weights='imagenet')

    model.summary()

    plot_model(model, to_file='model.png')

    return model

def resnet18_2d(image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):

    # Block 1
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(train_encoder, scale=False)
    bn_params = get_bn_params(train_encoder)
    conv_params = get_conv_params(train_encoder)
    init_filters = 64

    # INPUT
    inputs = layers.Input((image_rows, image_cols, input_channels))
    # resnet bottom
    x = layers.BatchNormalization(name='bn_data', **no_scale_bn_params)(inputs)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    skip_connection_1 = x
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # Stage 1, Unit 1 - Settings
    stage = 0
    block = 0
    strides = (1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    # defining shortcut connection
    shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 1, Unit 2 - Settings
    stage = 0
    block = 1
    strides = (1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 2, Unit 1 - Settings
    stage = 1
    block = 0
    strides=(2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_2 = x
    # defining shortcut connection
    shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 2, Unit 2 - Settings
    stage = 1
    block = 1
    strides = (1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 3, Unit 1 - Settings
    stage = 2
    block = 0
    strides=(2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_3 = x
    # defining shortcut connection
    shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 3, Unit 2 - Settings
    stage = 2
    block = 1
    strides = (1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 4, Unit 1 - Settings
    stage = 3
    block = 0
    strides=(2, 2)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 1 - Layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    skip_connection_4 = x
    # defining shortcut connection
    shortcut = layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
    # continue with convolution layers
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1_convpool', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Stage 4, Unit 2 - Settings
    stage = 3
    block = 1
    strides = (1, 1)
    filters = init_filters * (2 ** stage)
    conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)
    # Stage 1, Block 2 - Layers
    # defining shortcut connection
    shortcut = x
    # continue with convolution layers
    x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '1')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)
    x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
    x = layers.Activation('relu', name=relu_name + '2')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)
    x = layers.Add()([x, shortcut])

    # Resnet OUTPUT

    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)




    model = Model(inputs=[inputs], outputs=[x], name='resnet18')

    model.summary()

    plot_model(model, to_file='model.png')

    weights_path = utils.get_file('resnet18_imagenet_1000_no_top.h5.h5', WEIGHTS_PATH_NO_TOP,
        cache_subdir='models')

    model.load_weights(weights_path)

    return model

def resnet18_3d(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):

    # Block 1
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(train_encoder, scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = 64

    # INPUT
    inputs = layers.Input((image_depth, image_rows, image_cols, input_channels))
    # resnet bottom
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



    model = Model(inputs=[inputs], outputs=[x], name='resnet18')

    model.summary()

    plot_model(model, to_file='model.png')

    # weights_path = utils.get_file('resnet18_imagenet_1000_no_top.h5.h5', WEIGHTS_PATH_NO_TOP,
    #     cache_subdir='models')
    #
    # model.load_weights(weights_path)

    return model

if __name__ == '__main__':
    # vgg16_2d()
    # resnet18_2d_qubvel()
    resnet18_3d()