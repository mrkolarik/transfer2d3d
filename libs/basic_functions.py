# -------------------------------------  IMPORTS ----------------------------------

from __future__ import print_function
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.io import imread
import keras.backend as K
from libs.data2D import *
from libs.data3D import *
from models.unetvgg2d import *
import datetime
import keras.backend as K
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

smooth = 1.

#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#


# -------------------------------------  Dice coeficient LOSS FUNCTION ----------------------------------

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def predict():
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')

    imgs_test /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)

    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

    npy_mask_dir = 'test_mask_npy'
    if not os.path.exists(npy_mask_dir):
        os.mkdir(npy_mask_dir)

    np.save(os.path.join(npy_mask_dir, project_name + '_mask.npy'), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    imgs_mask_test = preprocess_squeeze(imgs_mask_test)
    # imgs_mask_test /= 1.7
    imgs_mask_test = np.around(imgs_mask_test, decimals=0)
    imgs_mask_test = (imgs_mask_test * 255.).astype(np.uint8)
    count_visualize = 1
    count_processed = 0
    count_processed = 0
    pred_dir = 'preds/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    pred_dir = os.path.join('preds/', project_name)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs_mask_test.shape[0]):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_mask_test[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0]))

    print('-' * 30)
    print('Prediction finished')
    print('-' * 30)


# -------------------------------------  Weights loading ----------------------------------

def load_model_weights(network_weights):
    try:
        generator = torch.load('./weights/{}.h5'.format(network_weights))
        print("\n--------model weights loaded--------\n")
    except:
        print("\n--------model weights not loaded--------\n")
        pass


# -------------------------------------  Check if folder exists, if not create ----------------------------------

def check_create_dir(dirpath):
    create_dir = os.path.join('./data/brain/', dirpath)
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = os.path.join('./data/chaos/', dirpath)
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/chaos/'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/chaos/2D'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/chaos/3D'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/brain/'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/brain/2D'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)
    create_dir = './results/brain/3D'
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)

# -------------------------------------  Load data train ----------------------------------

def load_data_train(dataset, dimension_mode, logger):
    if dataset == 'brain':
        if dimension_mode == "2D":
            dataset_dir = "./data/brain/numpy/2D"
            pred_dir = "./results/brain/2D"
            imgs_train, imgs_mask_train = load_train_data_2D()
        elif dimension_mode == "3D":
            dataset_dir = "./data/brain/numpy/3D"
            pred_dir = "./results/brain/3D"
            imgs_train, imgs_mask_train = load_train_data_3D()
    elif dataset == "chaos":
        if dimension_mode == "2D":
            dataset_dir = "./data/chaos/numpy/2D"
            pred_dir = "./results/chaos/2D"
        elif dimension_mode == "3D":
            dataset_dir = "./data/chaos/numpy/3D"
            pred_dir = "./results/chaos/3D"

    logger.write("\nnumber of training images: " + str(len(imgs_train)))

    print(' ---------------- Loading data done -----------------')

    return (imgs_train, imgs_mask_train)


# -------------------------------------  Load data test ----------------------------------

def load_data_test(dataset, dimension_mode, logger):
    if dataset == 'brain':
        if dimension_mode == "2D":
            dataset_dir = "./data/brain/numpy/2D"
            pred_dir = "./results/brain/2D"
            imgs_test = load_test_data_2D()
        elif dimension_mode == "3D":
            dataset_dir = "./data/brain/numpy/3D"
            pred_dir = "./results/brain/3D"
            imgs_test = load_test_data_3D()
    elif dataset == "chaos":
        if dimension_mode == "2D":
            dataset_dir = "./data/chaos/numpy/2D"
            pred_dir = "./results/chaos/2D"
        elif dimension_mode == "3D":
            dataset_dir = "./data/chaos/numpy/3D"
            pred_dir = "./results/chaos/3D"

    # -------------------------------------  Logger ----------------------------------
    logger.write("\nnumber of testing images: " + str(len(imgs_test)))
    print(' ---------------- Loading data done -----------------')

    return imgs_test