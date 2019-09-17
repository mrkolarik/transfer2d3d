# --------------------- IMPORTS ---------------------------

from __future__ import print_function
import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.io import imread


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

# -------------------------------------  Create Train data function ----------------------

def create_train_data_2D(data_path, image_rows, image_cols):
    train_data_path = os.path.join(data_path, 'original/train/')
    mask_data_path = os.path.join(data_path, 'original/masks_train/')
    dirs = os.listdir(train_data_path)

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    imgs_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load train images to numpy 2d array -------------
    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i] = img
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} 2d images'.format(i, count))

    imgs = imgs_temp

    print('Loading of train data done.')

    # ----------------------- Load mask images to numpy 2d array -------------

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(mask_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for mask_name in images:
            img_mask = imread(os.path.join(dirr, mask_name), as_grey=True)
            img_mask = img_mask.astype(np.uint8)
            img_mask = np.array([img_mask])
            imgs_mask_temp[i] = img_mask
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} mask 2d images'.format(i, count))

    imgs_mask = imgs_mask_temp

    print('Loading of masks done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs_mask = preprocess_2D(imgs_mask)
    imgs = preprocess_2D(imgs)

    print('Preprocessing of masks done.')

    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/2D/imgs_train.npy'), imgs)
    np.save(os.path.join(data_path, 'numpy/2D/imgs_mask_train.npy'), imgs_mask)

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    imgs = preprocess_squeeze_2D(imgs)
    imgs_mask = preprocess_squeeze_2D(imgs_mask)

    # ----------------------- Visualise training and mask images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/2D/train/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 256):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} train images'.format(count_processed, 500))

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/2D/masks/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 256):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} train images'.format(count_processed, 500))


    print('Saving to .npy files done.')

# -------------------------------------  Loading train data function ----------------------

def load_train_data_2D():
    imgs_train = np.load('./data/brain/numpy/2D/imgs_train.npy')
    imgs_mask_train = np.load('./data/brain/numpy/2D/imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

# -------------------------------------  Create test data function ----------------------

def create_test_data_2D(data_path, image_rows, image_cols):
    test_data_path = os.path.join(data_path, 'original/test/')
    dirs = os.listdir(test_data_path)
    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load all test images to numpy array -------------

    i = 0
    j = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for dirr in sorted(os.listdir(test_data_path)):
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)

            img = np.array([img])
            imgs[i] = img
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} test 2d images'.format(i, count))

    print('Loading done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs = preprocess_2D(imgs)

    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/2D/imgs_test.npy'), imgs)

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    imgs = preprocess_squeeze_2D(imgs)


    # ----------------------- Visualise test images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/2D/test/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]))

    print('Saving to .npy files done.')

# -------------------------------------  Create Mask test numpy files for testing purposes ----------------------

def create_mask_test_data_2D(data_path, image_rows, image_cols):
    test_data_path = os.path.join(data_path, 'original/masks_test/')
    dirs = os.listdir(test_data_path)
    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load all mask test images to numpy array -------------

    i = 0
    j = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for dirr in sorted(os.listdir(test_data_path)):
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)

            img = np.array([img])
            imgs[i] = img
            i += 1
            if (i % 100) == 0:
                print('Done: {0}/{1} test 2d images'.format(i, count))

    print('Loading done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs = preprocess_2D(imgs)

    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/2D/imgs_mask_test.npy'), imgs)

    print('Saving to .npy files done - mask test images 2D')

# -------------------------------------  Loading test data function ----------------------

def load_test_data_2D():
    imgs_test = np.load('./data/brain/numpy/2D/imgs_test.npy')
    return imgs_test

# -------------------------------------  Loading mask test data function ----------------------

def load_mask_test_data_2D():
    imgs_test = np.load('./data/brain/numpy/2D/imgs_mask_test.npy')
    return imgs_test

# ----------------------- Function to add dimension to arrays to be complied with keras dimension requirements
# ----------------------- Basically we add channel dimension -------------

def preprocess_2D(imgs):
    imgs = np.expand_dims(imgs, axis=3)
    print(' ---------------- preprocessed -----------------')
    return imgs


# ----------------------- Function to remove dimension from arrays to easier image generation
# ----------------------- Basically we remove channel dimension (last dimension) -------------

def preprocess_squeeze_2D(imgs):
    imgs = np.squeeze(imgs, axis=3)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


