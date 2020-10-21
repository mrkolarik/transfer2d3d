from keras.applications.vgg16 import VGG16

from libs.basic_functions import *
from matplotlib import pyplot
from models import Unet_vgg, resnet18, Unet_resnet18, Unet_original
from keras import utils

np.random.seed(1337)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.FATAL)
tf.set_random_seed(1337)


#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

# -------------------------------------  Generate weights for 3D unet with 2D VGG autoencoder ------------------------

def generate_planar_3D_weights_vgg(model_name):
    weights_filename = model_name
    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    vgg2d = VGG16(weights='imagenet')

    unetvgg3d = Unet_vgg.planar_3D_vgg()

    visualise_filters3d(unetvgg3d.get_layer('block1_conv2'))

    for layerunet in unetvgg3d.layers:
        for layervgg in vgg2d.layers:
            if (layerunet.name == layervgg.name and "conv" in layerunet.name):
                print(layerunet.name)
                filters, biases = layervgg.get_weights()
                filters = np.expand_dims(filters, axis=0)
                print(np.shape(np.expand_dims(filters, axis=0)))
                layerunet.set_weights([filters, biases])

    visualise_filters3d(unetvgg3d.get_layer('block1_conv2'))

    unetvgg3d.save_weights(os.path.join(weight_dir, weights_filename + '.h5'))

    print('-' * 30)
    print('Saved model weights to disk to file: ' + weight_dir + weights_filename + '.h5')
    print('-' * 30)

# -------------------------------------  Generate weights for 3D unet with 2D Resnet autoencoder ------------------------

def generate_planar_3D_weights_resnet(model_name):
    weights_filename = 'unet_resnet18_encoder_pseudo3d'
    weights_filename = model_name
    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    weights_path = utils.get_file('resnet18_imagenet_1000_no_top.h5.h5', 'https://github.com/qubvel/classification_models/'
                'releases/download/0.0.1/'
                'resnet18_imagenet_1000_no_top.h5',
                                  cache_subdir='models')

    model_2d = resnet18.resnet18_2d()

    model_2d.load_weights(weights_path)

    model_3d = Unet_resnet18.res_unet_resnet18()

    visualise_filters3d_nobias(model_3d.get_layer('stage1_unit1_conv2'))

    for layer3d in model_3d.layers:
        for layer2d in model_2d.layers:
            # Expand weights of conv layers
            if (layer2d.name == layer3d.name and "conv" in layer3d.name and "convpool" not in layer3d.name):
                print(layer3d.name)
                filters = layer2d.get_weights()[0]
                filters = np.expand_dims(filters, axis=0)
                # print(np.shape(filters))
                layer3d.set_weights([filters])
            # Expand weights of shortcut conv layers
            if (layer2d.name == layer3d.name and "_sc" in layer3d.name):
                print(layer3d.name)
                filters = layer2d.get_weights()[0]
                filters = np.expand_dims(filters, axis=0)
                # print(np.shape(filters))
                layer3d.set_weights([filters])
            # Expand weights of convpool layers
            elif (layer2d.name == layer3d.name and "convpool" in layer3d.name):
                print(layer3d.name)
                filters = layer2d.get_weights()[0]
                filters = np.stack([filters, filters])
                # print(np.shape(filters))
                layer3d.set_weights([filters])
            # Expand weights of rest of the layers
            elif (layer2d.name == layer3d.name and "conv" not in layer3d.name and "_sc" not in layer3d.name  and "convpool" not in layer3d.name):
                print(layer3d.name)
                # print(layer2d.name)
                layer3d.set_weights(layer2d.get_weights())

    # visualise_filters3d(unetvgg2d.layers[2])

    visualise_filters3d_nobias(model_3d.get_layer('stage1_unit1_conv2'))

    model_3d.save_weights(os.path.join(weight_dir, weights_filename + '.h5'))

    print('-' * 30)
    print('Saved model weights to disk to file: ' + weight_dir + weights_filename + '.h5')
    print('-' * 30)


# -------------------------------------  Generate initialized 3D resunet autoencoder weights------------------------

def generate_resunet_3D_weights(model_name, img_depth, img_rows, img_cols):
    weights_filename = 'unet_original_encoder'
    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    unetvgg3d = Unet_original.res_unet(16, 256, 256)

    unetvgg3d.save_weights(os.path.join(weight_dir, weights_filename + '.h5'))

    print('-' * 30)
    print('Saved model weights to disk to file: ' + weight_dir + weights_filename + '.h5')
    print('-' * 30)



# -------------------------------------  Visualise kernels of saved weights of models with bias -------

def visualise_model_weights(model_name, img_depth, img_rows, img_cols, layer_name):
    weights_filename = 'denseunetvgg3d_encoder'
    weights_filename_trained = 'denseunet3d_brain_brain_notpretrained_best'
    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    unetvgg3d_original = initialize_model(model_name, False, img_depth, img_rows, img_cols, False)
    unetvgg3d_original_trained = initialize_model(model_name, False, img_depth, img_rows, img_cols, False)

    load_model_weights(unetvgg3d_original, weights_filename)
    load_model_weights(unetvgg3d_original_trained, weights_filename_trained)

    visualise_filters3d(unetvgg3d_original.get_layer(layer_name))

    visualise_filters3d(unetvgg3d_original_trained.get_layer(layer_name))

    print('-' * 30)
    print('Visualised kernels of the selected layer')
    print('-' * 30)

# -------------------------------------  Visualise kernels of saved weights of models without bias -------

def visualise_model_weights_nobias(model_name, img_depth, img_rows, img_cols, layer_name):
    weights_filename = 'denseunetvgg3d_test'
    weights_filename_trained = 'denseunetvgg3dnobias_original'
    weight_dir = './weights/'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    unetvgg3d_original = initialize_model(model_name, False, img_depth, img_rows, img_cols, False)
    unetvgg3d_original_trained = initialize_model(model_name, False, img_depth, img_rows, img_cols, False)

    load_model_weights(unetvgg3d_original, weights_filename)
    load_model_weights(unetvgg3d_original_trained, weights_filename_trained)

    visualise_filters3d_nobias(unetvgg3d_original.get_layer(layer_name))

    visualise_filters3d_nobias(unetvgg3d_original_trained.get_layer(layer_name))

    print('-' * 30)
    print('Visualised kernels of the selected layer')
    print('-' * 30)

# -------------------------------------  Visualise few filters from selected layer --------------------------------
def visualise_filters3d(layer):
    # retrieve weights from the second hidden layer
    filters, biases = layer.get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    # print(layer.name)
    # print(np.shape(filters))
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, :, i]
        # plot each channel separately
        for j in range(0, f.shape[0]):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[j, :, :, 0], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()


# -------------------------------------  Visualise few filters from selected layer --------------------------------
def visualise_filters3d_nobias(layer):
    # retrieve weights from the second hidden layer
    filters = layer.get_weights()[0]
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    # print(layer.name)
    # print(np.shape(filters))
    n_filters, ix = 6, 1
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, :, i]
        # plot each channel separately
        for j in range(0, f.shape[0]):
            # specify subplot and turn of axis
            ax = pyplot.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(f[j, :, :, 0], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()


#                  __  __    _    ___ _   _
#                 |  \/  |  / \  |_ _| \ | |
#  _____   _____  | |\/| | / _ \  | ||  \| |  _____   _____
# |_____| |_____| | |  | |/ ___ \ | || |\  | |_____| |_____|
#                 |_|  |_/_/   \_\___|_| \_|
#


if __name__ == '__main__':
    generate_planar_3D_weights_vgg('planar_3d_vgg')

