from __future__ import print_function
import os
import numpy as np
from scipy.ndimage.filters import median_filter
from skimage.io import imsave, imread

# ------------------------------ GLOBAL VARIABLES ------------------------

data_path = './'

image_rows = int(256)
image_cols = int(256)
image_depth = 16

validation_percent = 8

validation_split = 100 // validation_percent

#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

def create_train_data():
    train_data_path = os.path.join(data_path, 'data/train_scans/')
    mask_data_path = os.path.join(data_path, 'data/train_masks/')
    numpy_data_path = os.path.join(data_path, 'data/numpy/')
    dirs = os.listdir(train_data_path)

    total = 0

    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    array_depth = 0
    j = 0
    for x in range(0, total):
        j += 1
        if j % (image_depth/2) == 0:
            j = 0
            array_depth += 1

    # array_depth = (total//image_depth)*2 +1

    imgs = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)

    imgs_validation = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_validation_mask = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)

    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        print(dirr)
        dirr = os.path.join(train_data_path, dirr)
        if not os.path.isdir(dirr):
            continue
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
        if x % validation_split == 0:
            imgs_validation = np.append(imgs_validation, np.expand_dims(np.append(imgs_temp[x], imgs_temp[x+1], axis=0), axis=0), axis=0)
        else:
            imgs = np.append(imgs, np.expand_dims(np.append(imgs_temp[x], imgs_temp[x+1], axis=0), axis=0), axis=0)


    print('Loading of train data done.')

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr = os.path.join(mask_data_path, dirr)
        if not os.path.isdir(dirr):
            continue
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
                    print('Done: {0}/{1} mask 3d images'.format(i, count))

    for x in range(0, imgs_mask_temp.shape[0]-1):
        if x % validation_split == 0:
            imgs_validation_mask = np.append(imgs_validation_mask,
                                  np.expand_dims(np.append(imgs_mask_temp[x], imgs_mask_temp[x + 1], axis=0), axis=0),
                                  axis=0)
        else:
            imgs_mask = np.append(imgs_mask,
                                  np.expand_dims(np.append(imgs_mask_temp[x], imgs_mask_temp[x + 1], axis=0), axis=0),
                                  axis=0)

    print('Loading of masks done.')

    count_processed = 0

    if not os.path.exists(numpy_data_path):
        os.mkdir(numpy_data_path)

    imgs_mask = preprocess(imgs_mask)
    imgs = preprocess(imgs)
    imgs_validation_mask = preprocess(imgs_validation_mask)
    imgs_validation = preprocess(imgs_validation)



    np.save(os.path.join(numpy_data_path, 'imgs_train.npy'), imgs)
    np.save(os.path.join(numpy_data_path, 'imgs_mask_train.npy'), imgs_mask)
    np.save(os.path.join(numpy_data_path, 'imgs_validation.npy'), imgs_validation)
    np.save(os.path.join(numpy_data_path, 'imgs_mask_validation.npy'), imgs_validation_mask)

    imgs = preprocess_squeeze(imgs)
    imgs_mask = preprocess_squeeze(imgs_mask)
    imgs_validation = preprocess_squeeze(imgs_validation)
    imgs_validation_mask = preprocess_squeeze(imgs_validation_mask)

    print('Saving numpy files done.')

    print('Saving visualised train files for control purposes:')

    pred_dir = 'data/visualised'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    count_processed = 0
    pred_dir = 'data/visualised/train_scans'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, min(imgs.shape[0], 30)):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, str('{:04}'.format(count_processed)) + '.png'), imgs[x][y], check_contrast=False)
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, 500))

    print('Saving visualised mask files for control purposes:')

    count_processed = 0
    pred_dir = 'data/visualised/train_masks'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, min(imgs_mask.shape[0], 30)):
        for y in range(0, imgs_mask.shape[1]):
            imsave(os.path.join(pred_dir, str('{:04}'.format(count_processed)) + '.png'), imgs_mask[x][y], check_contrast=False)
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} mask images'.format(count_processed, 500))

    print('Saving visualised validation files for control purposes:')

    count_processed = 0
    pred_dir = 'data/visualised/validation_train'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs_validation.shape[0]):
        for y in range(0, imgs_validation.shape[1]):
            imsave(os.path.join(pred_dir, str('{:04}'.format(count_processed)) + '.png'), imgs_validation[x][y], check_contrast=False)
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} validation images'.format(count_processed, 500))

    print('Saving visualised validation mask files for control purposes:')

    count_processed = 0
    pred_dir = 'data/visualised/validation_mask'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs_validation_mask.shape[0]):
        for y in range(0, imgs_validation_mask.shape[1]):
            imsave(os.path.join(pred_dir, str('{:04}'.format(count_processed)) + '.png'), imgs_validation_mask[x][y], check_contrast=False)
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} validation mask images'.format(count_processed, 500))


    print('Saving to .npy files done.')

def create_test_data():
    test_data_path = os.path.join(data_path, 'data/test_scans/')
    numpy_data_path = os.path.join(data_path, 'data/numpy/')
    dirs = os.listdir(test_data_path)

    total = 0

    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    array_depth = 0
    j = 0
    for x in range(0, total):
        j += 1
        if j % (image_depth/2) == 0:
            j = 0
            array_depth += 1

    print(array_depth)

    imgs = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    j = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    i = 0
    for dirr in sorted(os.listdir(test_data_path)):
        j = 0
        dirr = os.path.join(test_data_path, dirr)
        if not os.path.isdir(dirr):
            continue
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
        imgs = np.append(imgs, np.expand_dims(np.append(imgs_temp[x], imgs_temp[x + 1], axis=0), axis=0), axis=0)

    print(imgs.shape)

    print('Loading done.')

    imgs = preprocess(imgs)

    np.save(os.path.join(numpy_data_path, 'imgs_test.npy'), imgs)
    # np.save('imgs_test.npy', imgs)

    imgs = preprocess_squeeze(imgs)

    pred_dir = 'data/visualised'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    count_visualize = 1
    count_processed = 0
    pred_dir = 'data/visualised/test_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    for x in range(0, imgs.shape[1] // 4):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'), imgs[0][x], check_contrast=False)
        count_processed += 1
        # print("Saved beginning of the image file")
    # -------------------------------------  Print middle depth/2 images from each array ----------------------
    for x in range(0, imgs.shape[0]):
        for y in range(0, imgs.shape[1]):
            if (count_visualize > imgs.shape[1] // 4) and \
                    (count_visualize < ((imgs.shape[1] // 4) * 3 + 1)):
                imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'),
                       imgs[x][y], check_contrast=False)
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(
                        count_processed,
                        (imgs.shape[0] * imgs.shape[1]) // 2 + imgs.shape[1] // 2))

            count_visualize += 1
            if count_visualize == (imgs.shape[1] + 1):
                count_visualize = 1

    # -------------------------------------  Print last depth/4 images from last array -----------------------
    for y in range(((imgs.shape[1] // 4) * 3), imgs.shape[1]):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'),
               imgs[imgs.shape[0] - 1][y], check_contrast=False)
        count_processed += 1

    print('Saving to .npy files done.')

def create_reference_data():
    test_data_path = os.path.join(data_path, 'data/test_masks/')
    dirs = os.listdir(test_data_path)

    total = 0

    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    array_depth = 0
    j = 0
    for x in range(0, total):
        j += 1
        if j % (image_depth/2) == 0:
            j = 0
            array_depth += 1

    print(array_depth)

    imgs = np.ndarray((0, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_temp = np.ndarray((array_depth, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    j = 0
    print('-'*30)
    print('Creating reference images...')
    print('-'*30)

    i = 0
    for dirr in sorted(os.listdir(test_data_path)):
        j = 0
        dirr = os.path.join(test_data_path, dirr)
        if not os.path.isdir(dirr):
            continue
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
        imgs = np.append(imgs, np.expand_dims(np.append(imgs_temp[x], imgs_temp[x + 1], axis=0), axis=0), axis=0)

    print(imgs.shape)

    print('Loading done.')

    count_visualize = 1
    count_processed = 0
    pred_dir = './data/evaluation'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    # for x in range(0, 30):
    #     for y in range(0, imgs.shape[1]):
    #         imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
    #         count_processed += 1
    #         if (count_processed % 100) == 0:
    #             print('Done: {0}/{1} test images'.format(count_processed, imgs.shape[0]*imgs.shape[1]))

    for x in range(0, imgs.shape[1] // 4):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'), imgs[0][x], check_contrast=False)
        count_processed += 1
        # print("Saved beginning of the image file")
    # -------------------------------------  Print middle depth/2 images from each array ----------------------
    for x in range(0, imgs.shape[0]):
        for y in range(0, imgs.shape[1]):
            if (count_visualize > imgs.shape[1] // 4) and \
                    (count_visualize < ((imgs.shape[1] // 4) * 3 + 1)):
                imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'),
                       imgs[x][y], check_contrast=False)
                count_processed += 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} reference images'.format(
                        count_processed,
                        (imgs.shape[0] * imgs.shape[1]) // 2 + imgs.shape[1] // 2))

            count_visualize += 1
            if count_visualize == (imgs.shape[1] + 1):
                count_visualize = 1
    # -------------------------------------  Print last depth/4 images from last array -----------------------
    for y in range(((imgs.shape[1] // 4) * 3), imgs.shape[1]):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:04}") + '.png'),
               imgs[imgs.shape[0] - 1][y], check_contrast=False)
        count_processed += 1

    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('./data/numpy/imgs_test.npy')
    return imgs_test

def load_train_data():
    imgs_train = np.load('./data/numpy/imgs_train.npy')
    imgs_mask_train = np.load('./data/numpy/imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def load_validation_data():
    imgs_valid = np.load('./data/numpy/imgs_validation.npy')
    imgs_mask_valid = np.load('./data/numpy/imgs_mask_validation.npy')
    return imgs_valid, imgs_mask_valid

def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs

def visualise_predicitons(imgs, project_name):

    # ------------------------------------  Calculate output array dimensions -----------------------------

    imgs_mask_test = preprocess_squeeze(imgs)

    count_visualize = 1
    count_processed = 0

    for x in range(0, imgs_mask_test.shape[1] // 4):
        count_processed += 1
    # -------------------------------------  Print middle depth/2 images from each array ----------------------
    for x in range(0, imgs_mask_test.shape[0]):
        for y in range(0, imgs_mask_test.shape[1]):
            if (count_visualize > imgs_mask_test.shape[1] // 4) and \
                    (count_visualize < ((imgs_mask_test.shape[1] // 4) * 3 + 1)):
                count_processed += 1
            count_visualize += 1
            if count_visualize == (imgs_mask_test.shape[1] + 1):
                count_visualize = 1
    # -------------------------------------  Print last depth/4 images from last array -----------------------
    for y in range(((imgs_mask_test.shape[1] // 4) * 3), imgs_mask_test.shape[1]):
        count_processed += 1

    # ----------------------- Create output image array --------------------

    imgs_pred_output = np.zeros((count_processed, 256, 256))

    count_visualize = 1
    count_processed = 0

    print(np.shape(imgs_pred_output))
    print(np.shape(imgs_mask_test[0][0]))

    for x in range(0, imgs_mask_test.shape[1] // 4):
        imgs_pred_output[count_processed] = imgs_mask_test[0][x]
        count_processed += 1
    # -------------------------------------  Print middle depth/2 images from each array ----------------------
    for x in range(0, imgs_mask_test.shape[0]):
        for y in range(0, imgs_mask_test.shape[1]):
            if (count_visualize > imgs_mask_test.shape[1] // 4) and \
                    (count_visualize < ((imgs_mask_test.shape[1] // 4) * 3 + 1)):
                imgs_pred_output[count_processed] = imgs_mask_test[x][y]
                count_processed += 1
            count_visualize += 1
            if count_visualize == (imgs_mask_test.shape[1] + 1):
                count_visualize = 1
    # -------------------------------------  Print last depth/4 images from last array -----------------------
    for y in range(((imgs_mask_test.shape[1] // 4) * 3), imgs_mask_test.shape[1]):
        imgs_pred_output[count_processed] = imgs_mask_test[imgs_mask_test.shape[0] - 1][y]
        count_processed += 1

    print(np.shape(imgs_pred_output))

    # ----------------------- Normalize between 0 - 255 and apply median filter --------------------

    # pred_dir = 'preds/'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # pred_dir = os.path.join('preds/', project_name + '_nofilter')
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)



    # # ----------------------- Create pred images --------------------
    # count_processed = 0
    #
    # for x in range(0, imgs_pred_output.shape[0]):
    #     imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_pred_output[x],
    #            check_contrast=False)
    #     count_processed += 1
    #     if (count_processed % 100) == 0:
    #         print('Done: {0}/{1} test images'.format(count_processed, imgs_pred_output.shape[0]))

    # print('-' * 30)
    # print('Prediction nofilter finished')
    # print('-' * 30)

    imgs_pred_output = (imgs_pred_output * 255.).astype(np.uint8)


    threshold = imgs_pred_output.max() // 16
    imgs_pred_output[imgs_pred_output < threshold] = 0
    imgs_pred_output[imgs_pred_output > threshold] = 255

    pred_dir = 'preds/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    pred_dir = os.path.join('preds/', project_name)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    # ----------------------- Create pred images --------------------
    count_processed = 0

    for x in range(0, imgs_pred_output.shape[0]):
        imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_pred_output[x],
               check_contrast=False)
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, imgs_pred_output.shape[0]))

    print('-' * 30)
    print('Prediction threshold finished')
    print('-' * 30)


    # imgs_pred_output = median_filter(imgs_pred_output, mode='nearest', size=(2, 2, 2))
    #
    # imgs_pred_output = (imgs_pred_output * 255.).astype(np.uint8)
    #
    # threshold = imgs_pred_output.max() // 16
    # imgs_pred_output[imgs_pred_output < threshold] = 0
    # imgs_pred_output[imgs_pred_output > threshold] = 255
    #
    # pred_dir = 'preds/'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    # pred_dir = os.path.join('preds/', project_name + '_medianfilter')
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)
    #
    # # ----------------------- Create pred images --------------------
    # count_processed = 0
    #
    # for x in range(0, imgs_pred_output.shape[0]):
    #     imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_pred_output[x],
    #            check_contrast=False)
    #     count_processed += 1
    #     if (count_processed % 100) == 0:
    #         print('Done: {0}/{1} test images'.format(count_processed, imgs_pred_output.shape[0]))

    print('-' * 30)
    print('Prediction finished')
    print('-' * 30)



if __name__ == '__main__':
    # create_train_data()
    create_reference_data()
    create_test_data()

