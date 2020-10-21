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

# -------------------------------------  Create Load test image function ----------------------
def create_train_data(dataset, validation_split):
    if dataset == "volumetry":
        data_path = "./data/volumetry/"
    elif dataset == "segmentation":
        data_path = "./data/segmentation/"
    elif dataset == "denoising":
        data_path = "./data/denoising/"
    elif dataset == "detection":
        data_path = "./data/detection/"

    train_data_path = os.path.join(data_path, 'original/train/')

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    # ----------------------- Check input image for neccesary information - dtype and size -------------

    visualise_number = 256

    image_rows = 0
    image_cols = 0
    image_dtype = ''

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            image_dtype = img.dtype
            image_rows = img.shape[0]
            image_cols = img.shape[1]

            print(img.dtype)
            print(np.shape(img))

            break
        break

    validation_split = int(total * (1 - validation_split))

    print('Training set will contain ' + str(validation_split) + ' images, validation set will contain '
          + str(total - validation_split) + ' images.')

    print('-' * 30)
    print('Input train image of type ' + str(image_dtype) + ' with pixels in x dimension: '
          + str(image_cols) + ' and pixels in y dimension: ' + str(image_rows))
    print('-' * 30)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)

    # ----------------------- Load train images to numpy 2d array -------------
    i = 0
    print('-' * 30)
    print('Loading training images to numpy array')
    print('-' * 30)
    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            imgs[i] = imread(os.path.join(dirr, image_name), as_grey=True)
            i += 1
            if (i % 500) == 0:
                print('Done: {0}/{1} 2d images'.format(i, count))

    print('Maximum and minumum values of loaded data before scaling')

    print(imgs.max())
    print(imgs.min())

    imgs = imgs.astype('float32')

    imgs /= imgs.max()  # scale train images to [0, 1]

    print('Maximum and minumum values of loaded data after scaling')

    print(imgs.max())
    print(imgs.min())

    np.save(os.path.join(data_path, 'numpy/imgs_train.npy'), imgs[0:validation_split])
    np.save(os.path.join(data_path, 'numpy/imgs_train_valid.npy'), imgs[validation_split:])

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    imgs *= 255.  # scale train images to [0, 1]
    imgs = imgs.astype('uint8')

    print('Maximum and minumum values of loaded data before preprocessing')

    print(imgs.max())
    print(imgs.min())

    # ----------------------- Visualise training  images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/train/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, visualise_number):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} train images'.format(count_processed, visualise_number))

    # ----------------------- Create mask arrays !!!! -------------
    # ----------------------- Create mask arrays !!!! -------------
    # ----------------------- Create mask arrays !!!! -------------
    # ----------------------- Create mask arrays !!!! -------------

    masks_data_path = os.path.join(data_path, 'original/masks_train/')

    print('-' * 30)
    print('Creating mask images...')
    print('-' * 30)

    total = 0
    for root, dirs, files in os.walk(masks_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(masks_data_path)):
        dirr = os.path.join(masks_data_path, dirr)
        images = sorted(os.listdir(dirr))
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            image_dtype = img.dtype
            image_rows = img.shape[0]
            image_cols = img.shape[1]

            print(img.dtype)
            print(np.shape(img))

            break
        break

    print('-' * 30)
    print('Input mask image of type ' + str(image_dtype) + ' with pixels in x dimension: '
          + str(image_cols) + ' and pixels in y dimension: ' + str(image_rows))
    print('-' * 30)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)

    # ----------------------- Load train images to numpy 2d array -------------
    i = 0
    print('-' * 30)
    print('Loading training images to numpy array')
    print('-' * 30)
    for dirr in sorted(os.listdir(masks_data_path)):
        dirr = os.path.join(masks_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            imgs[i] = imread(os.path.join(dirr, image_name), as_grey=True)
            i += 1
            if (i % 500) == 0:
                print('Done: {0}/{1} 2d mask images'.format(i, count))

    print('Maximum and minumum values of loaded data before scaling')

    print(imgs.max())
    print(imgs.min())

    imgs = imgs.astype('float32')

    imgs /= imgs.max()  # scale train images to [0, 1]

    print('Maximum and minumum values of loaded data after scaling')

    print(imgs.max())
    print(imgs.min())

    np.save(os.path.join(data_path, 'numpy/imgs_mask_train.npy'), imgs[0:validation_split])
    np.save(os.path.join(data_path, 'numpy/imgs_mask_train_valid.npy'), imgs[validation_split:])

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    imgs *= 255.  # scale train images to [0, 1]
    imgs = imgs.astype('uint8')

    print('Maximum and minumum values of loaded data before preprocessing')

    print(imgs.max())
    print(imgs.min())

    # ----------------------- Visualise training and mask images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/masks/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, visualise_number):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 100) == 0:
            print('Done: {0}/{1} mask images'.format(count_processed, visualise_number))

    # ----------------------- Create test arrays !!!! -------------
    # ----------------------- Create test arrays !!!! -------------
    # ----------------------- Create test arrays !!!! -------------
    # ----------------------- Create test arrays !!!! -------------

    test_data_path = os.path.join(data_path, 'original/test/')

    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)

    total = 0
    for root, dirs, files in os.walk(test_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(test_data_path)):
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            image_dtype = img.dtype
            image_rows = img.shape[0]
            image_cols = img.shape[1]

            print(img.dtype)
            print(np.shape(img))

            break
        break

    print('-' * 30)
    print('Input test image of type ' + str(image_dtype) + ' with pixels in x dimension: '
          + str(image_cols) + ' and pixels in y dimension: ' + str(image_rows))
    print('-' * 30)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)

    # ----------------------- Load train images to numpy 2d array -------------
    i = 0
    print('-' * 30)
    print('Loading training images to numpy array')
    print('-' * 30)
    for dirr in sorted(os.listdir(test_data_path)):
        dirr = os.path.join(test_data_path, dirr)
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            imgs[i] = imread(os.path.join(dirr, image_name), as_grey=True)
            i += 1
            if (i % 500) == 0:
                print('Done: {0}/{1} 2d test images'.format(i, count))

    print('Maximum and minumum values of loaded data before scaling')

    print(imgs.max())
    print(imgs.min())

    imgs = imgs.astype('float32')

    imgs /= imgs.max()  # scale train images to [0, 1]

    print('Maximum and minumum values of loaded data after scaling')

    print(imgs.max())
    print(imgs.min())

    np.save(os.path.join(data_path, 'numpy/imgs_test.npy'), imgs)

    # ----------------------- Remove dimension to visualise part of the loaded images -------------

    imgs *= 255.  # scale train images to [0, 1]
    imgs = imgs.astype('uint8')

    print('Maximum and minumum values of loaded data before preprocessing')

    print(imgs.max())
    print(imgs.min())

    # ----------------------- Visualise training and mask images to preprocess folder -------------

    count_processed = 0
    pred_dir = os.path.join(data_path, 'preprocessed/test/')
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, visualise_number):
        imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x])
        count_processed += 1
        if (count_processed % 200) == 0:
            print('Done: {0}/{1} test images'.format(count_processed, visualise_number))


# -------------------------------------  Loading train data function ----------------------

def load_train_data(dataset):
    if dataset == "segmentation":
        dataset_dir = "./data/segmentation/numpy/"
    elif dataset == "denoising":
        dataset_dir = "./data/denoising/numpy/"
    elif dataset == "detection":
        dataset_dir = "./data/detection/numpy/"
    elif dataset == "volumetry":
        dataset_dir = "./data/volumetry/numpy/"

    imgs_train = np.load(os.path.join(dataset_dir, 'imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(dataset_dir, 'imgs_mask_train.npy'))
    return imgs_train, imgs_mask_train


# -------------------------------------  Loading train data function ----------------------

def load_train_valid_data(dataset):
    if dataset == "segmentation":
        dataset_dir = "./data/segmentation/numpy/"
    elif dataset == "denoising":
        dataset_dir = "./data/denoising/numpy/"
    elif dataset == "detection":
        dataset_dir = "./data/detection/numpy/"
    elif dataset == "volumetry":
        dataset_dir = "./data/volumetry/numpy/"

    imgs_train = np.load(os.path.join(dataset_dir, 'imgs_train_valid.npy'))
    imgs_mask_train = np.load(os.path.join(dataset_dir, 'imgs_mask_train_valid.npy'))
    return imgs_train, imgs_mask_train


def count_masks_positive(dataset):
    if dataset == "segmentation":
        dataset_dir = "./data/segmentation/numpy/"
    elif dataset == "denoising":
        dataset_dir = "./data/denoising/numpy/"
    elif dataset == "detection":
        dataset_dir = "./data/detection/numpy/"
    elif dataset == "volumetry":
        dataset_dir = "./data/volumetry/numpy/"

    train_data_path = os.path.join(data_path, 'original/masks_train/')

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    # ----------------------- Check input image for neccesary information - dtype and size -------------

    visualise_number = 256

    count_all_pixel = 400 * 400 * 256
    count_positive_pixel = 0
    image_dtype = ''

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(train_data_path)):
        path_to_dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(path_to_dirr))
        for image_name in images:
            count_positive_pixel += np.count_nonzero(imread(os.path.join(path_to_dirr, image_name), as_grey=True))
        print('Scan {0} contains {1} positive pixels which is {2} percent of all pixel values on mask'
              .format(dirr, count_positive_pixel, 100 * float(count_positive_pixel) / float(count_all_pixel)))
        count_positive_pixel = 0


def sliding_window(mask_array, image_array, cube_size):
    if mask_array.ndim != 3:
        print("Input shape must be a 3D array")
        return
    stride = cube_size // 2

    count_processed = 0

    voxel_limit = cube_size ** 3 // 25

    cube_data_path = "./data/segmentation/"

    mask_output_array = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')
    img_output_array = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')

    for x in range(0, (mask_array.shape[0] - cube_size) // stride):
        for y in range(0, (mask_array.shape[1] - cube_size) // stride):
            for z in range(0, (mask_array.shape[2] - cube_size) // stride):
                mask_cube = mask_array[x * stride:(x * stride + cube_size),
                               y * stride:(y * stride + cube_size), z * stride:(z * stride + cube_size)]
                img_cube = image_array[x * stride:(x * stride + cube_size),
                               y * stride:(y * stride + cube_size), z * stride:(z * stride + cube_size)]

                if np.count_nonzero(img_cube) > voxel_limit:
                    # pred_dir = os.path.join(cube_data_path, 'preprocessed/masks/')
                    # for visualise_image in range(0, mask_cube.shape[0]):
                    #     imsave(os.path.join(pred_dir, 'mask' + str(mask_output_array.shape[0]) + '_'
                    #                         + str(count_processed) + '.png'),
                    #            mask_cube[visualise_image], check_contrast=False)
                    #     imsave(os.path.join(pred_dir, 'image' + str(mask_output_array.shape[0]) + '_'
                    #                         + str(count_processed) + '.png'),
                    #            img_cube[visualise_image], check_contrast=False)
                    #     count_processed += 1
                    mask_output_array = np.append(mask_output_array, np.expand_dims(mask_cube, axis=0), axis=0)
                    img_output_array = np.append(img_output_array, np.expand_dims(img_cube, axis=0), axis=0)
                    count_processed += 1
                    if(count_processed % 1000 == 0):
                        print("Created {0} brain samples".format(count_processed))
                    # print(current_cube.shape)
                    # print(output_array.shape)
            # print(y)
        # print(x)
        # print(mask_output_array.shape)
    print(mask_output_array.shape)

    # print(float(np.count_nonzero(mask_output_array)) / float(mask_output_array.shape[0] * cube_size ** 3))

    return img_output_array, mask_output_array


def create_cubes_train(dataset, cube_size):
    if dataset == "segmentation":
        dataset_dir = "./data/segmentation/numpy/"
    elif dataset == "denoising":
        dataset_dir = "./data/denoising/numpy/"
    elif dataset == "detection":
        dataset_dir = "./data/detection/numpy/"
    elif dataset == "volumetry":
        dataset_dir = "./data/volumetry/numpy/"

    train_data_path = os.path.join(data_path, 'original/train/')
    masks_data_path = os.path.join(data_path, 'original/masks_train/')

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    # ----------------------- Check input image for neccesary information - dtype and size -------------

    visualise_number = 256
    total = 256

    image_rows = 0
    image_cols = 0
    image_dtype = ''

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            image_dtype = img.dtype
            image_rows = img.shape[0]
            image_cols = img.shape[1]

            print(img.dtype)
            print(np.shape(img))

            break
        break

    total = 256
    print("Training folder contains {0} images".format(total))
    # validation_split = int(total * (1-validation_split))

    imgs = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)
    masks = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)

    mask_cubes = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')
    img_cubes = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')

    i = 0

    print('-' * 30)
    print('Loading training images to numpy array')
    print('-' * 30)
    for dirr in sorted(os.listdir(train_data_path)):
        dirr_train = os.path.join(train_data_path, dirr)
        dirr_masks = os.path.join(masks_data_path, dirr)
        images = sorted(os.listdir(dirr_train))
        count = total
        for image_name in images:
            imgs[i] = imread(os.path.join(dirr_train, image_name), as_grey=True)
            masks[i] = imread(os.path.join(dirr_masks, image_name), as_grey=True)
            i += 1
            if (i % 1000) == 0:
                print('Done loading {0}/{1} 2d images'.format(i, count))
        img_cubes_temp, mask_cubes_temp = sliding_window(masks, imgs, cube_size)
        img_cubes = np.append(img_cubes, img_cubes_temp, axis=0)
        mask_cubes = np.append(mask_cubes, mask_cubes_temp, axis=0)
        img_cubes_temp = None
        mask_cubes_temp = None
        i = 0
        print("Created brain samples for scan {0}".format(dirr))
        print("Currently created {0} brain cube samples".format(img_cubes.shape))


    print("Loaded images in numpy array are of shape {0}".format(imgs.shape))
    print("Loaded masks in numpy array are of shape {0}".format(masks.shape))

    # img_cubes, mask_cubes = sliding_window(masks, imgs, cube_size)

    img_cubes = img_cubes.astype('float16')
    mask_cubes = mask_cubes.astype('float16')

    img_cubes /= max(img_cubes.max(), 255)  # scale train images to [0, 1]
    mask_cubes /= max(mask_cubes.max(), 255)  # scale train images to [0, 1]

    img_cubes *= 255  # scale train images to [0, 255]
    mask_cubes *= 255  # scale train images to [0, 255]

    print('Maximum and minimum values of loaded data after scaling:')

    print(img_cubes.max())
    print(img_cubes.min())

    np.save('./data/brain_cube_samples/img_cubes.npy', img_cubes.astype("uint8"))
    np.save('./data/brain_cube_samples/mask_cubes.npy', mask_cubes.astype("uint8"))

    print('-' * 30)
    print('Creating cube sample images done!')
    print('-' * 30)

def create_cubes_test(dataset, cube_size):
    if dataset == "segmentation":
        dataset_dir = "./data/segmentation/numpy/"
    elif dataset == "denoising":
        dataset_dir = "./data/denoising/numpy/"
    elif dataset == "detection":
        dataset_dir = "./data/detection/numpy/"
    elif dataset == "volumetry":
        dataset_dir = "./data/volumetry/numpy/"

    train_data_path = os.path.join(data_path, 'original/test/')
    masks_data_path = os.path.join(data_path, 'original/masks_test/')

    print('-' * 30)
    print('Creating testing images...')
    print('-' * 30)

    # ----------------------- Check input image for neccesary information - dtype and size -------------

    visualise_number = 256
    total = 256

    image_rows = 0
    image_cols = 0
    image_dtype = ''

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(train_data_path)):
        dirr = os.path.join(train_data_path, dirr)
        images = sorted(os.listdir(dirr))
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_grey=True)
            image_dtype = img.dtype
            image_rows = img.shape[0]
            image_cols = img.shape[1]

            print(img.dtype)
            print(np.shape(img))

            break
        break

    total = 256
    print("Training folder contains {0} images".format(total))
    # validation_split = int(total * (1-validation_split))

    imgs = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)
    masks = np.ndarray((total, image_rows, image_cols), dtype=image_dtype)

    mask_cubes = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')
    img_cubes = np.ndarray((0, cube_size, cube_size, cube_size), dtype='float32')

    i = 0

    print('-' * 30)
    print('Loading training images to numpy array')
    print('-' * 30)
    for dirr in sorted(os.listdir(train_data_path)):
        dirr_train = os.path.join(train_data_path, dirr)
        dirr_masks = os.path.join(masks_data_path, dirr)
        images = sorted(os.listdir(dirr_train))
        count = total
        for image_name in images:
            imgs[i] = imread(os.path.join(dirr_train, image_name), as_grey=True)
            masks[i] = imread(os.path.join(dirr_masks, image_name), as_grey=True)
            i += 1
            if (i % 1000) == 0:
                print('Done loading {0}/{1} 2d images'.format(i, count))
        img_cubes_temp, mask_cubes_temp = sliding_window(masks, imgs, cube_size)
        img_cubes = np.append(img_cubes, img_cubes_temp, axis=0)
        mask_cubes = np.append(mask_cubes, mask_cubes_temp, axis=0)
        img_cubes_temp = None
        mask_cubes_temp = None
        i = 0
        print("Created brain samples for scan {0}".format(dirr))
        print("Currently created {0} brain cube samples".format(img_cubes.shape))


    print("Loaded images in numpy array are of shape {0}".format(imgs.shape))
    print("Loaded masks in numpy array are of shape {0}".format(masks.shape))

    # img_cubes, mask_cubes = sliding_window(masks, imgs, cube_size)

    img_cubes = img_cubes.astype('float16')
    mask_cubes = mask_cubes.astype('float16')

    img_cubes /= max(img_cubes.max(), 255)  # scale train images to [0, 1]
    mask_cubes /= max(mask_cubes.max(), 255)  # scale train images to [0, 1]

    img_cubes *= 255  # scale train images to [0, 255]
    mask_cubes *= 255  # scale train images to [0, 255]

    print('Maximum and minimum values of loaded data after scaling:')

    print(img_cubes.max())
    print(img_cubes.min())

    np.save('./data/brain_cube_samples/img_cubes_test.npy', img_cubes.astype("uint8"))
    np.save('./data/brain_cube_samples/mask_cubes_test.npy', mask_cubes.astype("uint8"))

    print('-' * 30)
    print('Creating cube sample images done!')
    print('-' * 30)

def create_segmentation_dataset(cube_size):
    data_path = "./data/segmentation/"
    img_cubes, mask_cubes = create_cubes("segmentation", cube_size)
    imgs_train = np.load(os.path.join(dataset_dir, 'imgs_train.npy'))
    imgs_mask_train = np.load(os.path.join(dataset_dir, 'imgs_mask_train.npy'))

def skulstrip_images(dataset):
    if dataset == "skullstripped":
        data_path = "./data/skullstripped/"

    train_data_path = os.path.join(data_path, 'scans/')
    masks_data_path = os.path.join(data_path, 'masks/')
    stripped_data_path = os.path.join(data_path, 'stripped/')

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)

    # ----------------------- Check input image for neccesary information - dtype and size -------------

    total = 0
    for root, dirs, files in os.walk(train_data_path):
        total += len(files)

    for dirr in sorted(os.listdir(train_data_path)):
        train_dirr = os.path.join(train_data_path, dirr)
        mask_dirr = os.path.join(masks_data_path, dirr)
        stripped_dirr = os.path.join(stripped_data_path, dirr)
        if not os.path.exists(stripped_dirr):
            os.mkdir(stripped_dirr)
        images = sorted(os.listdir(train_dirr))
        for image_name in images:
            original_img = imread(os.path.join(train_dirr, image_name), as_grey=True)
            mask_img = imread(os.path.join(mask_dirr, image_name), as_grey=True)
            mask_img = mask_img//255
            stripped_image = original_img*mask_img
            imsave(os.path.join(stripped_dirr, image_name), stripped_image, check_contrast=False)
        print("Finished scan {0}".format(dirr))
