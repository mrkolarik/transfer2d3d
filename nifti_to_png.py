import os

import nibabel as nib
import numpy as np
import scipy.ndimage
from skimage.io import imsave

# ------------------------------ GLOBAL VARIABLES ------------------------


output_image_size = [256, 256, 256]


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#


def image_histogram_equalization(image, number_bins=1024):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 65535 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def convert_scan_data():  # Function to load nifti files and convert them to normalized Pngs for later use
    data_path = './msseg/Unprocessed training dataset/TrainingDataset_MSSEG'
    resampled_path = './data/png/scans/'
    spline_order = 2
    i = 0
    print('-' * 30)
    print('Creating training images...')
    for dirr in sorted(os.listdir(data_path)):  # Iterate through directories
        dirr_path = os.path.join(data_path, dirr)
        if not os.path.isdir(dirr_path):
            continue
        nifti_files = sorted(os.listdir(dirr_path))
        count = 0
        for nifti in nifti_files:  # Load each nifti file
            if nifti == '3DFLAIR.nii.gz':
                n1_img = nib.load(os.path.join(dirr_path, nifti))
                numpy_nii = np.array(n1_img.dataobj)
                numpy_nii = numpy_nii.astype('float16')
                numpy_nii = np.rot90(numpy_nii, k=3, axes=(2, 1))
                numpy_nii = np.flip(numpy_nii, 0)  # Files are correctly rotated
                # print('-' * 30)
                # print('Loaded nifti file {0} from directory {1} with min values {2} and max value {3} in shape {4} '
                #       'which is ready for normalizing'.format(nifti, dirr, numpy_nii.min(), numpy_nii.max(),
                #                                               numpy_nii.shape))
                # print('Loaded nifti file {0} from directory {1} has mean value {2} and '
                #       'std value {3}'.format(nifti, dirr, numpy_nii.mean(), numpy_nii.std()))
                # print('-' * 30)
                # print('Normalizing')
                # print('-' * 30)
                numpy_nii = 65535 * (numpy_nii - numpy_nii.min()) / (numpy_nii.max() - numpy_nii.min())
                numpy_nii = scipy.ndimage.zoom(numpy_nii, (output_image_size[0] / numpy_nii.shape[0],
                                                           output_image_size[1] / numpy_nii.shape[1],
                                                           output_image_size[2] / numpy_nii.shape[2]), order=spline_order)

                numpy_nii = 65535 * (numpy_nii - numpy_nii.min()) / (numpy_nii.max() - numpy_nii.min())
                # print('Loaded nifti file {0} from directory {1} with min values {2} and max value {3} in shape {4} '
                #       'which is ready for normalizing'.format(nifti, dirr, numpy_nii.min(), numpy_nii.max(),
                #                                               numpy_nii.shape))
                # print('Loaded nifti file {0} from directory {1} has mean value {2} and '
                #       'std value {3}'.format(nifti, dirr, numpy_nii.mean(), numpy_nii.std()))
                pred_dir = os.path.join(resampled_path, dirr)
                if not os.path.exists(pred_dir):
                    os.mkdir(pred_dir)
                count_processed = 0
                # print('-' * 30)
                # print('Saving resampled images')
                # print('-' * 30)
                for x in range(0, numpy_nii.shape[0]):
                    imsave(os.path.join(pred_dir, str(f"{(count_processed + 1):04}") + '.png'),
                           (numpy_nii[x] / 256).astype("uint8"), check_contrast=False)
                    count_processed += 1
                print('-' * 30)
                print('Done nifti file {0} visualised images'.format(dirr))


def convert_masks_ms_data():  # Function to load nifti files and convert them to normalized Pngs for later use
    data_path = './msseg/Unprocessed training dataset/TrainingDataset_MSSEG'
    resampled_path = './data/png/masks/'
    spline_order = 0
    i = 0
    print('-' * 30)
    print('Creating training images...')
    for dirr in sorted(os.listdir(data_path)):  # Iterate through directories
        dirr_path = os.path.join(data_path, dirr)
        if not os.path.isdir(dirr_path):
            continue
        nifti_files = sorted(os.listdir(dirr_path))
        count = 0
        for nifti in nifti_files:  # Load each nifti file
            if nifti == 'Consensus.nii.gz':
                n1_img = nib.load(os.path.join(dirr_path, nifti))
                numpy_nii = np.array(n1_img.dataobj)
                numpy_nii = numpy_nii.astype('uint8')
                numpy_nii = np.rot90(numpy_nii, k=3, axes=(2, 1))
                numpy_nii = np.flip(numpy_nii, 0)  # Files are correctly rotated
                # print('-' * 30)
                # print('Loaded nifti file {0} from directory {1} with min values {2} and max value {3} in shape {4} '
                #       'which is ready for normalizing'.format(nifti, dirr, numpy_nii.min(), numpy_nii.max(),
                #                                               numpy_nii.shape))
                # print('-' * 30)
                # print('Resampling')
                # print('-' * 30)
                numpy_nii = 255 * numpy_nii
                numpy_nii = scipy.ndimage.zoom(numpy_nii, (output_image_size[0] / numpy_nii.shape[0],
                                                           output_image_size[1] / numpy_nii.shape[1],
                                                           output_image_size[2] / numpy_nii.shape[2]), order=spline_order)
                # print('Loaded nifti file {0} from directory {1} with min values {2} and max value {3} in shape {4} '
                #       'which is ready for normalizing'.format(nifti, dirr, numpy_nii.min(), numpy_nii.max(),
                #                                               numpy_nii.shape))
                pred_dir = os.path.join(resampled_path, dirr)
                if not os.path.exists(pred_dir):
                    os.mkdir(pred_dir)
                count_processed = 0
                # print('-' * 30)
                # print('Saving resampled images')
                # print('-' * 30)
                for x in range(0, numpy_nii.shape[0]):
                    imsave(os.path.join(pred_dir, str(f"{(count_processed + 1):04}") + '.png'),
                           (numpy_nii[x]).astype("uint8"), check_contrast=False)
                    count_processed += 1
                print('-' * 30)
                print('Done nifti file {0} visualised images'.format(dirr))
                # print(" ")


if __name__ == '__main__':
    convert_scan_data()
    convert_masks_ms_data()