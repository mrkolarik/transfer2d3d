# -------------------------------------  IMPORTS ----------------------------------

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
from libs.model_functions import *

np.random.seed(1337)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.FATAL)
tf.set_random_seed(1337)

# -------------------------------------  LOGGER INIT ----------------------------------

logger = open("./logs/log.txt", "a+")
logger.write("\n\n" + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

# -------------------------------------  PARSER CODE - DELETE IF NOT NEEDED------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--network",type=str,default="ode",help="choose between fusionnet & unet")
# parser.add_argument("--batch_size",type=int,default=1,help="batch size")
# parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
# args = parser.parse_args()
# -------------------------------------  HYPER PARAMETERS ----------------------------------

batch_size = 1
num_gpu = 1
learning_rate = 1e-5
lr_decay = 0.000000199
epoch = 600
# OPTIONS: - setting optimizer: adam, radam
optimizer = 'radam'

# -------------------------------------  MODEL PARAMETERS ----------------------------------

K.set_image_data_format('channels_last')

# OPTIONS - choose a model: unetvgg2d,
network = 'unetvgg2d'
# OPTIONS - choose datasets: brain, chaos
dataset = 'brain'
# OPTIONS - choose weights initialize: expandedvgg,
weights = 'expandedvgg'
# OPTIONS: - loading of different data is set: 2D, 3D,
dimension_mode = '3D'

# Enables training procedure weights
train = True
# Enables prediction procedure weights
predict = True
# Enables loading weights
load_weights = True
# Prints model summary and saves model graph in model directory
print_model = True

# ------------------------------------- Set image size based on dataset -------------------------

image_depth_3D = 16

if dataset == 'brain':
    image_rows = int(400)
    image_cols = int(400)
    if dimension_mode == "2D":
        image_depth = 1
    elif dimension_mode == "3D":
        image_depth = image_depth_3D
elif dataset == "chaos":
    image_rows = int(256)
    image_cols = int(256)
    if dimension_mode == "2D":
        image_depth = 1
    elif dimension_mode == "3D":
        image_depth = image_depth_3D

# -------------------------------------  LOGGER ----------------------------------

logger.write("\nNETWORK: " + network + " with loaded weights: " + weights)
logger.write("\ndimension_mode: " + str(dimension_mode))
logger.write("\nbatch_size: " + str(batch_size))
logger.write("\nimg_size: " + str(image_rows))
logger.write("\nepochs: " + str(epoch))
logger.write("\noptimizer: " + str(optimizer))


#                  __  __    _    ___ _   _
#                 |  \/  |  / \  |_ _| \ | |
#  _____   _____  | |\/| | / _ \  | ||  \| |  _____   _____
# |_____| |_____| | |  | |/ ___ \ | || |\  | |_____| |_____|
#                 |_|  |_/_/   \_\___|_| \_|
#

if __name__ == '__main__':
    # model = initialize_model(network, print_model, image_rows, image_cols)
    # compile_model(model, optimizer, learning_rate, lr_decay,)
    # if load_weights:
    #     load_model_weights(weights)
    # if train:
    #     train_network(model, network, batch_size, epoch, *(load_data_train(dataset, dimension_mode, logger)))
    # if predict:
    #     predict_results(model, batch_size, load_test_data_2D())
    generate_png_images(load_mask_test_data_3D(), dataset, dimension_mode, network)
    logger.close()
    # predict()
