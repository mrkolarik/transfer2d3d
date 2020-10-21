"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers, Model, utils
from keras import backend as K


K.set_image_data_format('channels_last')

import os

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def vgg16_2d(image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
    inputs = layers.Input((image_rows, image_cols, input_channels))
    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x_output = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(inputs=[inputs], outputs=[x_output], name='vgg16')

    model.summary()

    weights_path = utils.get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP,
        cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc')

    model.load_weights(weights_path)

    return model

def vgg16_3d(image_depth = 16, image_rows = 256, image_cols = 256, input_channels = 3, train_encoder = True):
    inputs = layers.Input((image_depth, image_rows, image_cols, input_channels))
    # Block 1
    block1_conv1 = layers.Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = layers.Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block1_conv2')(block1_conv1)
    block1_pool = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    block2_conv1 = layers.Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv1')(block1_pool)
    block2_conv2 = layers.Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block2_conv2')(block2_conv1)
    block2_pool = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    block3_conv1 = layers.Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv1')(block2_pool)
    block3_conv2 = layers.Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv2')(block3_conv1)
    block3_conv3 = layers.Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block3_conv3')(block3_conv2)
    block3_pool = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(block3_conv3)

    # Block 4
    block4_conv1 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv1')(block3_pool)
    block4_conv2 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv2')(block4_conv1)
    block4_conv3 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block4_conv3')(block4_conv2)
    block4_pool =layers. MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(block4_conv3)

    # Block 5
    block5_conv1 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv1')(block4_pool)
    block5_conv2 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv2')(block5_conv1)
    block5_conv3 = layers.Conv3D(512, (1, 3, 3), activation='relu', padding='same', name='block5_conv3')(block5_conv2)
    x_output = layers.MaxPooling3D((1, 1, 1), strides=(1, 1, 1), name='block5_pool')(block5_conv3)

    model = Model(inputs=[inputs], outputs=[x_output], name='vgg16')

    model.summary()

    return model

if __name__ == '__main__':
    # vgg16_2d()
    vgg16_3d()