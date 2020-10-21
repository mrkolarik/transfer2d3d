import random
import os
import numpy as np
from scipy.ndimage.filters import median_filter
import os
import random

import numpy as np
from scipy.ndimage.filters import median_filter


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img

def data_gen_training_segmentation(data_type, batch_size, output_depth, output_rows, output_cols,
                      shuffle, augment, smoothing=False, validation_split = 0.1):

    data_path = "./data/segmentation/numpy/"

    if data_type == "train":
        train_input = np.load(os.path.join(data_path, 'imgs_train.npy'))
        train_input = train_input[0:validation_split]
        masks_input = np.load(os.path.join(data_path, 'imgs_mask_train.npy'))
        masks_input = masks_input[0:validation_split]
    elif data_type == "validation":
        train_input = np.load(os.path.join(data_path, 'imgs_train.npy'))
        train_input = train_input[validation_split:]
        masks_input = np.load(os.path.join(data_path, 'imgs_mask_train.npy'))
        masks_input = masks_input[validation_split:]

    train_return = np.zeros((batch_size, output_depth, output_rows, output_cols)).astype('float32')
    masks_return = np.zeros((batch_size, output_depth, output_rows, output_cols)).astype('float32')

    visualise_count = 0

    pred_dir_train = './results/generator_preprocessed/train'
    if not os.path.exists(pred_dir_train):
        os.mkdir(pred_dir_train)
    pred_dir_masks = './results/generator_preprocessed/masks'
    if not os.path.exists(pred_dir_masks):
        os.mkdir(pred_dir_masks)

    while (True):
        for train_counter in range(0, train_input.shape[0]):    # if shuffle=false then go through entire dataset

            # start_time = time.time()
            if shuffle:
                random_shuffle = random.randint(0, train_input.shape[0])
            else:
                random_shuffle = 0

            # Random numbers for data augmentation probability:
            rotate = random.uniform(0.0, 1.0)
            rotate_angle = random.randint(0, 4)
            add_noise = random.uniform(0.0, 1.0)


            image = train_input[(train_counter+depth_counter+random_shuffle)%train_input.shape[0]]
            mask = masks_input[(train_counter+depth_counter+random_shuffle)%masks_input.shape[0]]

            if(depth_counter==0 and data_type == "train" and select_masks==True and mask_select_probability < 0.3):
                mask_sum = np.sum(mask)+1
                mask_ratio = mask_sum / mask.shape[0] ** 2


                while(mask_ratio < mask_threshold):
                    random_shuffle = random.randint(0, train_input.shape[0])
                    image = train_input[(train_counter + depth_counter + random_shuffle) % train_input.shape[0]]
                    mask = masks_input[(train_counter + depth_counter + random_shuffle) % masks_input.shape[0]]
                    mask_sum = np.sum(mask)+1
                    mask_ratio = mask_sum / mask.shape[0] ** 2

            if augment:
                if data_type == 'train':
                    # Vertical mirroring
                    # if mirror_vertical < 0.2:
                    #     image = cv2.flip(image, 0)
                    #     mask = cv2.flip(mask, 0)
                    # # Horizontal mirroring
                    # if mirror_horizontal < 0.2:
                    #     image = cv2.flip(image, 1)
                    #     mask = cv2.flip(mask, 1)
                    # # Random rotation
                    if rotate < 0.2:
                        image = rotate_img(image, rotate_angle)
                        mask = rotate_img(mask, rotate_angle)
                        mask[mask > 0.5] = 1
                    # Shearing
                    if shear < 0.2:
                        image = shear_img(image, shear_angle)
                        mask = shear_img(mask, shear_angle)
                        mask[mask > 0.5] = 1
                    # Adding gaussian noise
                    if add_noise < 0.4:
                        image = gaussian_noise(image)

            # image = cv2.resize(image, (output_rows, output_cols), interpolation=cv2.INTER_CUBIC)
            # mask = cv2.resize(mask, (output_rows, output_cols), interpolation=cv2.INTER_NEAREST)

            if(blur==True and blur_number>0):
                mask = median_filter(mask, size=blur_number)

            # imsave(os.path.join(pred_dir_train, 'pre_processed_' + str(visualise_count) + '.png'), image)
            # imsave(os.path.join(pred_dir_masks, 'pre_processed_' + str(visualise_count) + '.png'), mask)
            # # # imsave(os.path.join(pred_dir_masks, 'pre_processed_' + str(visualise_count) + '.png'), cv2.resize(mask, (output_rows//16, output_cols//16), interpolation=cv2.INTER_NEAREST))
            # # #
            # visualise_count += 1
    #     #
    #     print(time.time()-start_time)

            # train_return[batch_counter][depth_counter] = (image-0.5)*255
            train_return[batch_counter][depth_counter] = image
            masks_return[batch_counter][depth_counter] = mask
            # masks_return_small[batch_counter][depth_counter] = cv2.resize(mask, (output_rows//2, output_cols//2), interpolation=cv2.INTER_NEAREST)

        yield np.expand_dims(train_return, axis=4), np.expand_dims(masks_return, axis=4)

def data_gen_testing(dataset_project, batch_size, output_depth, output_rows, output_cols):

    if dataset_project == "segmentation":
        data_path = "./data/segmentation/numpy/"
    elif dataset_project == "denoising":
        data_path = "./data/denoising/numpy/"
    elif dataset_project == "detection":
        data_path = "./data/detection/numpy/"
    elif dataset_project == "volumetry":
        data_path = "./data/volumetry/numpy/"

    pred_input = np.load(os.path.join(data_path, 'imgs_test.npy')).astype('float32')

    # scale between [0, 1]

    pred_input /= pred_input.max()

    pred_return = np.zeros((batch_size, output_depth, output_rows, output_cols)).astype('float32')

    # visualise_count = 0
    #
    # pred_dir_test = './results/generator_preprocessed/test'
    # if not os.path.exists(pred_dir_train):
    #     os.mkdir(pred_dir_test)


    while (True):
        for train_counter in range(0, pred_input.shape[0]):    # if shuffle=false then go through entire dataset
            pred_return[train_counter % batch_size] = pred_input[train_counter]
            # yield if the pred return is not 0, if it is filled with values and if the for cycle ends
            if(train_counter != 0 and train_counter % batch_size == 0 or train_counter == pred_input.shape[0]-1):
                yield np.expand_dims(pred_return, axis=4)
                pred_return.fill(0)