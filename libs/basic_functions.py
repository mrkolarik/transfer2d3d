from __future__ import print_function

import os
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

smooth = 1.


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#


# -------------------------------------  Weights loading ----------------------------------

def load_model_weights(model, network_weights):
    try:
        model.load_weights('./weights/{}.h5'.format(network_weights))
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

def load_data_train(dataset, dimension_mode, logger, size):
    if dataset == 'brain':
        if dimension_mode == "2D":
            dataset_dir = "./data/brain/numpy/2D"
            pred_dir = "./results/brain/2D"
            imgs_train, imgs_mask_train = load_train_data_2D(dataset, size)
        elif dimension_mode == "3D":
            dataset_dir = "./data/brain/numpy/3D"
            pred_dir = "./results/brain/3D"
            imgs_train, imgs_mask_train = load_train_data_3D(dataset, size)
    elif dataset == "chaos":
        if dimension_mode == "2D":
            dataset_dir = "./data/chaos/numpy/2D"
            pred_dir = "./results/chaos/2D"
            imgs_train, imgs_mask_train = load_train_data_2D(dataset, size)
        elif dimension_mode == "3D":
            dataset_dir = "./data/chaos/numpy/3D"
            pred_dir = "./results/chaos/3D"
            imgs_train, imgs_mask_train = load_train_data_3D(dataset, size)

    logger.write("\nnumber of training images: " + str(len(imgs_train)))

    print(' ---------------- Loading data done -----------------')

    return (imgs_train, imgs_mask_train)


# -------------------------------------  Load data test ----------------------------------

def load_data_test(dataset, dimension_mode, logger, size):
    if dataset == 'brain':
        if dimension_mode == "2D":
            dataset_dir = "./data/brain/numpy/2D"
            pred_dir = "./results/brain/2D"
            imgs_test = load_test_data_2D(dataset, size)
        elif dimension_mode == "3D":
            dataset_dir = "./data/brain/numpy/3D"
            pred_dir = "./results/brain/3D"
            imgs_test = load_test_data_3D(dataset, size)
    elif dataset == "chaos":
        if dimension_mode == "2D":
            dataset_dir = "./data/chaos/numpy/2D"
            pred_dir = "./results/chaos/2D"
            imgs_test = load_test_data_2D(dataset, size)
        elif dimension_mode == "3D":
            dataset_dir = "./data/chaos/numpy/3D"
            pred_dir = "./results/chaos/3D"
            imgs_test = load_test_data_3D(dataset, size)

    # -------------------------------------  Logger ----------------------------------
    logger.write("\nnumber of testing images: " + str(len(imgs_test)))
    print(' ---------------- Loading data done -----------------')

    return imgs_test


def set_image_size(dataset, cube_size, depth_3D):
    if dataset == "segmentation":
        image_depth = depth_3D
        image_rows = 400
        image_cols = 400
    else:
        image_depth = cube_size
        image_rows = cube_size
        image_cols = cube_size
    return image_depth, image_cols, image_rows
