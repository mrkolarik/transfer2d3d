import os
import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from data_load import load_train_data, load_test_data, visualise_predicitons, load_validation_data
from models import Unet_vgg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.set_random_seed(256)
np.random.seed(256)
K.set_image_data_format('channels_last')

# ------------------------------ GLOBAL VARIABLES ------------------------


project_name = 'resunet_vgg'
img_rows = 256
img_cols = 256
img_depth = 16
smooth = 1.


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def segmentation_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred) + 1


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    # Prepare train and validation data

    imgs_scans_train, imgs_mask_train = load_train_data()
    imgs_scans_valid, imgs_mask_valid = load_validation_data()
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_scans_train = imgs_scans_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_scans_train /= 128.  # scale input images to [0, 2]
    imgs_scans_train = imgs_scans_train-1.  # scale input images to [-1, 1]

    imgs_scans_valid = imgs_scans_valid.astype('float32')
    imgs_mask_valid = imgs_mask_valid.astype('float32')
    imgs_mask_valid /= 255.  # scale masks to [0, 1]
    imgs_scans_valid /= 128.  # scale input images to [0, 2]
    imgs_scans_valid = imgs_scans_valid-1.  # scale input images to [-1, 1]

    imgs_scans_train = np.repeat(imgs_scans_train, 3, axis=4)  # repeat three times before multi modality loading
    imgs_scans_valid = np.repeat(imgs_scans_valid, 3, axis=4)  # repeat three times before multi modality loading


    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = Unet_vgg.res_unet_vgg(image_depth=img_depth, image_rows=img_rows, image_cols=img_cols, train_encoder=False)

    model.compile(optimizer=Adam(lr=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199),
                  loss=segmentation_loss, metrics=['accuracy', dice_coef])

    model.summary()

    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_dice_coef',
                                       save_best_only=True, mode='max')

    # Load planar 3D encoder
    model.load_weights(os.path.join(weight_dir, 'planar_3d_vgg.h5'), by_name=True)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir, project_name + '.txt'), separator=',', append=False)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    model.fit(
        x=imgs_scans_train,
        y=imgs_mask_train,
        batch_size=3,
        epochs=100,
        callbacks=[model_checkpoint, csv_logger],
        shuffle=True,
        validation_data=(imgs_scans_valid, imgs_mask_valid),
    )

    print('-' * 30)
    print('Training finished')
    print('-' * 30)


def predict():
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')

    imgs_test /= 128.  # scale input images to [0, 2]
    imgs_test = imgs_test-1.  # scale input images to [-1, 1]

    imgs_test = np.repeat(imgs_test, 3, axis=4)

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)

    # ----------------------- define model ----------------------------------

    model = Unet_vgg.res_unet_vgg(image_depth=img_depth, image_rows=img_rows, image_cols=img_cols)

    model.summary()

    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)

    npy_mask_dir = './data/numpy/test_mask'
    if not os.path.exists(npy_mask_dir):
        os.mkdir(npy_mask_dir)

    np.save(os.path.join(npy_mask_dir, project_name + '_mask.npy'), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    visualise_predicitons(imgs_mask_test, project_name)

if __name__ == '__main__':
    # train()
    predict()
