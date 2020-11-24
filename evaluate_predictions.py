import numpy as np
import os
import glob
import cv2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# import SimpleITK as sitk
# from scipy import ndimage
# from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, jaccard_score

#                   __                  _   _
#                  / _|_   _ _ __   ___| |_(_) ___  _ __  ___
#  _____   _____  | |_| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  _____   _____
# |_____| |_____| |  _| |_| | | | | (__| |_| | (_) | | | \__ \ |_____| |_____|
#                 |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#

def evaluate(Vref,Vseg):
    dice=DICE(Vref,Vseg)
    ravd=RAVD(Vref,Vseg)
    acc = np.sum(Vref == Vseg)/(Vref.shape[0] * Vref.shape[1] ** 2)

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(Vseg == 1, Vref == 1))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(Vseg == 0, Vref == 0))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(Vseg == 1, Vref == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(Vseg == 0, Vref == 1))

    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    pos_pred_val = TP/ (TP+FP)
    neg_pred_val = TN/ (TN+FN)
    
    return dice, ravd, acc, specificity, sensitivity


def DICE(Vref,Vseg):
    dice=2*(Vref & Vseg).sum()/(Vref.sum() + Vseg.sum() + 0.00001)
    return dice

def RAVD(Vref,Vseg):
    ravd=(abs(Vref.sum() - Vseg.sum())/Vref.sum())*100
    return ravd

def png_series_reader(dir):
    V = []
    png_file_list=glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list:
        image = cv2.imread(filename,0)
        V.append(image)
    V = np.array(V,order='A')
    V = V/255
    V = np.around(V, decimals=0)
    V = V.astype(bool)
    return V

def evaluate_segmentation(ref_dir, pred_dir):
    print(png_series_reader(ref_dir).shape)
    # Calculate results for center 1
    dice, ravd, accuracy, specificity, sensitivity = evaluate(png_series_reader(ref_dir)[0:255], png_series_reader(pred_dir)[0:255])
    print("Calculated results for center 1:")
    # print('Calculated Accuracy score  :' + str(accuracy))
    print('Calculated DICE        ' + str(dice))
    print('Calculated Specificity ' + str(specificity))
    print('Calculated Sensitivity ' + str(sensitivity))
    # Calculate results for center 7
    dice, ravd, accuracy, specificity, sensitivity = evaluate(png_series_reader(ref_dir)[256:512], png_series_reader(pred_dir)[256:512])
    print("Calculated results for center 7:")
    # print('Calculated Accuracy score  :' + str(accuracy))
    print('Calculated DICE        ' + str(dice))
    print('Calculated Specificity ' + str(specificity))
    print('Calculated Sensitivity ' + str(sensitivity))
    # Calculate results for center 8
    dice, ravd, accuracy, specificity, sensitivity = evaluate(png_series_reader(ref_dir)[513:767], png_series_reader(pred_dir)[513:767])
    print("Calculated results for center 8:")
    # print('Calculated Accuracy score  :' + str(accuracy))
    print('Calculated DICE        ' + str(dice))
    print('Calculated Specificity ' + str(specificity))
    print('Calculated Sensitivity ' + str(sensitivity))

    dice, ravd, accuracy, specificity, sensitivity = evaluate(png_series_reader(ref_dir), png_series_reader(pred_dir))
    return dice, ravd, accuracy, specificity, sensitivity
    # return dice, ravd, accuracy, specificity, sensitivity

#                  __  __    _    ___ _   _
#                 |  \/  |  / \  |_ _| \ | |
#  _____   _____  | |\/| | / _ \  | ||  \| |  _____   _____
# |_____| |_____| | |  | |/ ___ \ | || |\  | |_____| |_____|
#                 |_|  |_/_/   \_\___|_| \_|
#

if __name__ == '__main__':
    pred_data_path = './preds/'
    for dirr in sorted(os.listdir(pred_data_path)):
        print('-' * 30)
        print("Evaluation results of experiment " + dirr + " : ")
        print('-' * 30)
        dirr = os.path.join(pred_data_path, dirr)
        dice, ravd, accuracy, specificity, sensitivity = evaluate_segmentation('./data/evaluation/', dirr)
        print("Calculated complete results:")
        # print('Calculated Accuracy score  :' + str(accuracy))
        print('Calculated DICE        ' + str(dice))
        print('Calculated Specificity ' + str(specificity))
        print('Calculated Sensitivity ' + str(sensitivity))
        # # print('Calculated RAVD ' + str(ravd))
        print('-' * 30)
