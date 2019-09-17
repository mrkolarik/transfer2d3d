from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from skimage.io import imsave
import numpy as np

np.random.seed(256)
import tensorflow as tf
tf.set_random_seed(256)

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K

from models.data2D import load_train_data, load_test_data, preprocess_squeeze

K.set_image_data_format('channels_last')

project_name = '2D-Unet'
img_rows = 400
img_cols = 400
img_depth = 1
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv12)

    conv21 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv22 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)

    conv31 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv32 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv31)
    conv33 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv32)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv33)

    conv41 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv42 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv41)
    conv43 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(conv42)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv43)

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv51)
    conv53 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv52)

    up6 = add([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv53), conv43])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = add([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv33])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = add([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv22])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = add([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv12])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)


    model = Model(inputs=[inputs], outputs=[conv10])

    # model.summary()
    # #plot_model(model, to_file='model.png')
    #
    # model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)


    imgs_train, imgs_mask_train = load_train_data()
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_train /= 255.  # scale masks to [0, 1]


    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])



    print('-'*30)
    print('Training finished')
    print('-'*30)

def predict():



    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = load_test_data()
    imgs_test = imgs_test.astype('float32')



    imgs_test /= 255.  # scale masks to [0, 1]


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

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

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    predict()
