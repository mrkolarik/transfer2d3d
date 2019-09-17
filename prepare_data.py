# -------------- IMPORTS -----------------
from __future__ import print_function
import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from skimage.io import imread

from libs.data2D import *
from libs.data3D import *
from libs.basic_functions import check_create_dir

# -------------------------------- Creating necessary folder structure -----------------------

check_create_dir('preprocessed/')
check_create_dir('preprocessed/2D')
check_create_dir('preprocessed/3D')
check_create_dir('numpy/')
check_create_dir('numpy/2D')
check_create_dir('numpy/3D')
check_create_dir('original/')

# -------------- SETTINGS -----------------

# OPTIONS: brain, chaos
dataset = 'brain'

if dataset == "isbi":
    dataset_dir = "./data/chaos/"
elif dataset == "brain":
    dataset_dir = "./data/brain/"
    image_rows = int(400)
    image_cols = int(400)

#Settings - 2D, 3D
dimension_mode = "3D"
image_depth = 16

if dataset == 'brain':
    if dimension_mode == "2D":
        create_train_data_2D(dataset_dir, image_rows, image_cols)
        create_test_data_2D(dataset_dir, image_rows, image_cols)
        create_mask_test_data_2D(dataset_dir, image_rows, image_cols)
    elif dimension_mode == "3D":
        create_train_data_3D(dataset_dir, image_rows, image_cols, image_depth)
        create_test_data_3D(dataset_dir, image_rows, image_cols, image_depth)
        create_mask_test_data_3D(dataset_dir, image_rows, image_cols, image_depth)