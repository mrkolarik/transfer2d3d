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

def create_train_data_3D(data_path, image_rows, image_cols, image_depth):
    train_data_path = os.path.join(data_path, 'original/train/')
    mask_data_path = os.path.join(data_path, 'original/masks_train/')
    dirs = os.listdir(train_data_path)

    visualise_number = 10

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    array_depth = (total//image_depth)*2

    imgs = np.ndarray((array_depth, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load train images to numpy 3D array -------------

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = array_depth
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i,j] = img
            j += 1
            if j % (image_depth/2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, count))

    for x in range(0, imgs_temp.shape[0]-1):
        imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

    del imgs_temp

    print('Loading of train data done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs = preprocess_3D(imgs)

    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/3D/imgs_train.npy'), imgs)

    print('Saving to .npy files done.')

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    print('Preprocessing images for visualisation.')

    imgs = preprocess_squeeze_3D(imgs)

    # ----------------------- Visualise training images to preprocess folder -------------


    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/3D/train/')

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, visualise_number):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: Saving {0}/{1} train preprocessed images'.format(count_processed, visualise_number*imgs.shape[1]))

    del imgs

    print('Saving preprocessed images done')

    # ----------------------- Load mask images to numpy 3D array -------------

    imgs_mask = np.ndarray((array_depth, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr = os.path.join(mask_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = array_depth
        for mask_name in images:
            img_mask = imread(os.path.join(dirr, mask_name), as_grey=True)
            img_mask = img_mask.astype(np.uint8)

            img_mask = np.array([img_mask])

            imgs_mask_temp[i,j] = img_mask

            j += 1
            if j % (image_depth/2) == 0:
                j = 0
                i += 1
                if (i % 100) == 0:
                    print('Done: Saving {0}/{1} preprocessed mask 3d images'.format(i, count))

    for x in range(0, imgs_mask_temp.shape[0]-1):
        imgs_mask[x]=np.append(imgs_mask_temp[x], imgs_mask_temp[x+1], axis=0)

    print('Loading of masks done.')

    del imgs_mask_temp

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs_mask = preprocess_3D(imgs_mask)

    print('Preprocessing of masks done.')

    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/3D/imgs_mask_train.npy'), imgs_mask)

    print('Saving to .npy files done.')

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    print('Preprocessing images for visualisation.')

    imgs_mask = preprocess_squeeze_3D(imgs_mask)

    # ----------------------- Visualise mask images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/3D/masks/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, visualise_number):
        for y in range(0, imgs_mask.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, visualise_number*imgs_mask.shape[1]))

    print('Saving preprocessed mask images done')

# -------------------------------------  Create  test numpy files ----------------------

def create_test_data_3D(data_path, image_rows, image_cols, image_depth):
    test_data_path = os.path.join(data_path, 'original/test/')
    dirs = os.listdir(test_data_path)

    visualise_number = 10

    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    array_depth = (total // image_depth) * 2


    imgs = np.ndarray((array_depth, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load all test images to numpy array -------------

    i = 0
    print('-'*30)
    print('Creating testing 3D images...')
    print('-'*30)
    for dirr in sorted(os.listdir(test_data_path)):
        j = 0
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = ((total//image_depth)*2)+image_depth
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i,j] = img
            j += 1
            if j % (image_depth/2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, count))

    for x in range(0, imgs_temp.shape[0]-1):
        imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

    print('Loading done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs = preprocess_3D(imgs)
    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/3D/imgs_test.npy'), imgs)

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    print('Preprocessing images for visualisation.')

    imgs = preprocess_squeeze_3D(imgs)

    # ----------------------- Visualise test images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/3D/test/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs.shape[0]):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done:Saving {0}/{1} preprocessed test images'.format(count_processed, imgs.shape[0]*imgs.shape[1]))

    print('Saving to .npy files done.')

# -------------------------------------  Create Mask test numpy files for testing purposes ----------------------

def create_mask_test_data_3D(data_path, image_rows, image_cols, image_depth):
    test_data_path = os.path.join(data_path, 'original/masks_test/')
    dirs = os.listdir(test_data_path)

    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    array_depth = (total // image_depth) * 2

    imgs = np.ndarray((array_depth, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    # ----------------------- Load all test images to numpy array -------------

    i = 0
    print('-'*30)
    print('Creating testing 3D images...')
    print('-'*30)
    for dirr in sorted(os.listdir(test_data_path)):
        j = 0
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = ((total//image_depth)*2)+image_depth
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            img = img.astype(np.uint8)
            img = np.array([img])
            imgs_temp[i,j] = img
            j += 1
            if j % (image_depth/2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, count))

    for x in range(0, imgs_temp.shape[0]-1):
        imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

    print('Loading done.')

    # ----------------------- Add dimension to arrays to be complied with keras dimension requirements -------------
    # ----------------------- Basically we add channel dimension -------------

    imgs = preprocess_3D(imgs)
    # ----------------------- Save loaded images to numpy files -------------

    np.save(os.path.join(data_path, 'numpy/3D/imgs_mask_test.npy'), imgs)

    print('Saving to .npy files done - mask test images 3D')

# -------------------------------------  Loading test data function ----------------------

def load_test_data_3D():
    imgs_test = np.load('./data/brain/numpy/3D/imgs_test.npy')
    return imgs_test

# -------------------------------------  Loading test data function ----------------------

def load_mask_test_data_3D():
    imgs_test = np.load('./data/brain/numpy/3D/imgs_mask_test.npy')
    return imgs_test

# -------------------------------------  Loading function for 3D train data ----------------------

def load_train_data_3D():
    imgs_train = np.load('./data/brain/numpy/3D/imgs_train.npy')
    imgs_mask_train = np.load('./data/brain/numpy/3D/imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


# ----------------------- Function to add dimension to arrays to be complied with keras dimension requirements
# ----------------------- Basically we add channel dimension -------------

def preprocess_3D(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

# ----------------------- Function to remove dimension from arrays to easier image generation
# ----------------------- Basically we remove channel dimension (last dimension) -------------

def preprocess_squeeze_3D(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs

