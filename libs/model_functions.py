# -------------------------------------  IMPORTS ----------------------------------
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from models.unetvgg2d import *
import datetime
import keras.backend as K
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
from libs.basic_functions import *
from libs.data2D import *
from libs.data3D import *
from keras_radam import RAdam


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

# -------------------------------------  Model initialization ----------------------------------

def initialize_model(network_name, print_model, image_rows, image_cols):
    if network_name == "unetvgg2d":
        model_initialize = get_unetvgg2d(image_rows, image_cols)
    else:
        print(' ---------------- Model not initialized - wrong name -----------------')

    if print_model:
        try:
            model_initialize.summary()
            plot_model(model_initialize, to_file='./models/{}.png'.format(network_name))
        except:
            pass

    print(' ---------------- Model initialized -----------------')

    return model_initialize


# -------------------------------------  Model compilation ----------------------------------

def compile_model(model_compile, model_optimizer, lr, decay,):
    if model_optimizer == "adam":
        model_compile.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay),
                                 loss=dice_coef_loss, metrics=['accuracy', dice_coef])
        print(' ---------------- Model compiled with ADAM optimizer-----------------')
    elif model_optimizer == "radam":
        model_compile.compile(RAdam(), loss=dice_coef_loss, metrics=['accuracy', dice_coef])
        print(' ---------------- Model compiled with RADAM optimizer-----------------')
    else:
        print(' ---------------- Model not compiled -----------------')


# -------------------------------------  Training function definition --------------------------------
def train_network(model, network_name, batch_size, epoch, imgs_train, imgs_mask_train):
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_train /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, network_name + '.h5'),
                                       monitor='val_loss', save_best_only=True)

    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir, network_name + '.txt'), separator=',', append=False)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epoch, verbose=1,
              shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])

    print('-' * 30)
    print('Training finished')
    print('-' * 30)


# -------------------------------------  Testing function definition --------------------------------
def predict_results(model, batch_size, imgs_test):
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    imgs_test = imgs_test.astype('float32')

    imgs_test /= 255.  # scale masks to [0, 1]

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, batch_size=batch_size, verbose=1)

    # imgs_mask_test /= 1.7
    imgs_mask_test = np.around(imgs_mask_test, decimals=0)
    imgs_mask_test = (imgs_mask_test * 255.).astype(np.uint8)

    return imgs_mask_test

# -------------------------------------  Generate PNGS from numpy arrays --------------------------------

def generate_png_images(input_numpy_array, dataset, dimension_mode, network_name):

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    if dataset == 'brain':
        if dimension_mode == "2D":
            pred_dir = "./results/brain/2D"
            imgs_mask_test = preprocess_squeeze_2D(input_numpy_array)
        elif dimension_mode == "3D":
            pred_dir = "./results/brain/3D"
            imgs_mask_test = preprocess_squeeze_3D(input_numpy_array)
    elif dataset == "chaos":
        if dimension_mode == "2D":
            pred_dir = "./results/chaos/2D"
            imgs_mask_test = preprocess_squeeze_2D(input_numpy_array)
        elif dimension_mode == "3D":
            pred_dir = "./results/chaos/3D"
            imgs_mask_test = preprocess_squeeze_3D(input_numpy_array)

    create_dir = os.path.join(pred_dir, (network_name + datetime.datetime.now().strftime("_%d_%m_%Y_%H:%M:%S")))
    if not os.path.exists(create_dir):
        os.mkdir(create_dir)

    pred_dir = create_dir

    count_visualize = 1
    count_processed = 0

    # -------------------------------------  Generate PNGS from 2D numpy arrays --------------------------------

    if dimension_mode == "2D":
        for x in range(0, imgs_mask_test.shape[0]):
            imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_mask_test[x])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0]))

    # -------------------------------------  Generate PNGS from 3D numpy arrays --------------------------------
    # -------------------------------------  Bit of spaghetti code, but works, sorry ... -----------------------

    elif dimension_mode == "3D":
        # -------------------------------------  Print first x (depht/4) imgs from first 3D array -----------------
        for x in range(0, imgs_mask_test.shape[1]//4):
            imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_mask_test[0][x])
            count_processed += 1
        # -------------------------------------  Print middle depth/2 images from each array ----------------------
        for x in range(0, imgs_mask_test.shape[0]-1):
            for y in range(0, imgs_mask_test.shape[1]):
                if (count_visualize > imgs_mask_test.shape[1]//4) and \
                        (count_visualize < ((imgs_mask_test.shape[1]//4)*3+1)):
                    imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'),
                           imgs_mask_test[x][y])
                    count_processed += 1
                count_visualize += 1
                if count_visualize == (imgs_mask_test.shape[1]+1):
                    count_visualize = 1
                if (count_processed % 100) == 0:
                    print('Done: {0}/{1} test images'.format(
                        count_processed,
                        (imgs_mask_test.shape[0] * imgs_mask_test.shape[1])//2+imgs_mask_test.shape[1]//2))
        # -------------------------------------  Print last depth/4 images from last array -----------------------
        for y in range(((imgs_mask_test.shape[1]//4)*3), imgs_mask_test.shape[1]):
            imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'),
                   imgs_mask_test[imgs_mask_test.shape[0]-1][y])
            count_processed += 1

    print('-' * 30)
    print('Prediction finished')
    print('-' * 30)