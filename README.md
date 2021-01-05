## Planar 3D Transfer Learning for End to End Unimodal MRI Unbalanced Data Segmentation
### Utilizing 2D pre-trained weights in 3D segmentation with application on multiple sclerosis lesion segmentation

Hello everyone, this is a repository containing code to Paper "Planar 3D Transfer Learning for End to End Unimodal MRI Unbalanced Data Segmentation" soon to be published (accepted) to ICPR 2020. Pre-print Arxiv version here: https://arxiv.org/abs/2011.11557

Most useful parts of this repository are python keras scripts with source code for generating planar 3D weights for 2D to 3D transfer learning.


## Overview

We present a novel approach of 2D to 3D transfer learning based on mapping pre-trained 2D convolutional neural network weights into planar 3D kernels.

<p align="center">
  <img width="300" src="img/planar3d.PNG"> <br>
  <b>Figure_1:</b> Visualisation of 2D → Planar 3D convolutional kernel transformation
</p>

The method is validated by the proposed planar 3D res-u-net network with encoder transferred from the 2D VGG-16, which is applied for a single-stage unbalanced 3D image data segmentation.

<p align="center">
  <img width="800" src="img/architecture.PNG"> <br>
  <b>Figure_2:</b> Architecture of the proposed planar 3D res-u-net with VGG-16 planar 3D encoder
</p>


## How to use
There are two steps to use our implementation in your own project:

1. Generate Planar 3D VGG weights by running this script - it will save Planar 3D VGG weights to the ./weights folder:
```bash
generate_weights.py
```

2. Then you can use our model Res-U-Net from the ./models/Unet_vgg with planar encoder in your Keras code as follows:
```bash
model = Unet_vgg.res_unet_vgg(image_depth=img_depth, image_rows=img_rows, image_cols=img_cols, train_encoder=False)
model.load_weights('./weights/planar_3d_vgg.h5', by_name=True)
```
This code instatiates the Res-U-Net Keras model and loads the Planar 3D VGG weights by name (only the VGG encoder layers weights will be loaded and the rest remains randomly initialized).


## Citation and references

Please cite our work as:

@article{kolarik2020planar,<br>
&nbsp;   &nbsp;    title={Planar 3D Transfer Learning for End to End Unimodal MRI Unbalanced Data Segmentation},<br>
&nbsp;   &nbsp;    author={Kolarik, Martin and Burget, Radim and Travieso-Gonzalez, Carlos M and Kocica, Jan},<br>
&nbsp;   &nbsp;    journal={arXiv preprint arXiv:2011.11557},<br>
&nbsp;   &nbsp;    year={2020}<br>
}


## Reproducing the paper results

1.  Get the data - here is a link to subscribe to the MSSEG 16 challenge, data will be available after manual verification of your team by organizers:
https://portal.fli-iam.irisa.fr/msseg-challenge/overview

2.  Download the Unprocessed training dataset and unzip it to the ./msseg folder in the root directory. The path to the first Flair scan should be from the root:
```bash
./msseg/Unprocessed training dataset/TrainingDataset_MSSEG/01016SACH/3DFLAIR.nii.gz
```

3.  Create and activate the virtual Python conda environment:
```bash
conda create --name planar3d --file environment.txt
conda activate planar3d
```
this command creates an Anaconda environment called "planar3d" and installs all the necessary packages. We were notified that conda has problems with installing the open-cv package. If this happens to you, you can install it via pip:
```bash
pip install opencv-python
```

4. Run the following script to generate the png processed dataset.:
```bash
nifti_to_png.py
```
 The conda environment can be created by running: 
```bash
conda create --name planar3d --file environment.txt
```
this command creates an Anaconda environment called "planar3d" and installs all the necessary packages. If there will be any packages missing, install them via pip and let us know, we will update the Readme.

5.  Divide the dataset to the training / test set by copying the folders from data/png/scans and data/png/masks to the corresponding folders data/train_scans - data/train_masks - data/test_scans - data/test_masks. The description of which scans were used as the test scans in each crossvalidation round can be found in the short acoompanying paper to our main ICPR publication. In the first round of crossvalidation the data was divided like this:

```bash
./data/
├── ./test_masks
│   ├── ./test_masks/01042GULE
│   ├── ./test_masks/07043SEME
│   └── ./test_masks/08037ROGU
├── ./test_scans
│   ├── ./test_scans/01042GULE
│   ├── ./test_scans/07043SEME
│   └── ./test_scans/08037ROGU
├── ./train_masks
│   ├── ./train_masks/01016SACH
│   ├── ./train_masks/01038PAGU
│   ├── ./train_masks/01039VITE
│   ├── ./train_masks/01040VANE
│   ├── ./train_masks/07001MOEL
│   ├── ./train_masks/07003SATH
│   ├── ./train_masks/07010NABO
│   ├── ./train_masks/07040DORE
│   ├── ./train_masks/08002CHJE
│   ├── ./train_masks/08027SYBR
│   ├── ./train_masks/08029IVDI
│   └── ./train_masks/08031SEVE
├── ./train_scans
│   ├── ./train_scans/01016SACH
│   ├── ./train_scans/01038PAGU
│   ├── ./train_scans/01039VITE
│   ├── ./train_scans/01040VANE
│   ├── ./train_scans/07001MOEL
│   ├── ./train_scans/07003SATH
│   ├── ./train_scans/07010NABO
│   ├── ./train_scans/07040DORE
│   ├── ./train_scans/08002CHJE
│   ├── ./train_scans/08027SYBR
│   ├── ./train_scans/08029IVDI
│   └── ./train_scans/08031SEVE
```

6.  Run the script: 
```bash
data_load.py 
```
to load the png files to numpy ready for neural network input.

7.  Download the weights from the 1st round of crossvalidation here:
https://drive.google.com/file/d/1Dq4Q6u0ghqAmiNcdFBvML7-fVzQMj2Hu/view?usp=sharing
and copy them to the ./weights directory

8.  Run the script:
```bash
original_paper_reproduction.py
```
to generate predictions

9.  Run the script:
```bash
evaluate_predictions.py
```
which should results in following output:

```bash
------------------------------
Evaluation results of experiment planar_cross_1 : 
------------------------------
(768, 256, 256)
Calculated results for center 1:
Calculated DICE        0.6313213001687128
Calculated Specificity 0.999973596266512
Calculated Sensitivity 0.4655478150728309
Calculated results for center 7:
Calculated DICE        0.6769804285243028
Calculated Specificity 0.999771726259143
Calculated Sensitivity 0.6048149205442954
Calculated results for center 8:
Calculated DICE        0.6704865676549892
Calculated Specificity 0.9984986197405552
Calculated Sensitivity 0.8287365515106169
Calculated complete results:
Calculated DICE        0.6582231144377745
Calculated Specificity 0.9994186303613052
Calculated Sensitivity 0.624245195852427
------------------------------
```
