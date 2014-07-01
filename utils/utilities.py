__author__ = 'adeb'

import os
import sys
import h5py
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
import nibabel as nib


def load_config(default_file):
    """
    Load a config file specified in the command line.
    Attributes:
        default_file (string): If no *args is provided, use this argument.
    """
    if len(sys.argv) == 1:
        cf = __import__(default_file)
    else:
        cf = __import__(str(sys.argv[1]))

    return cf


def create_directories(folder_name):
    # Create directories if they don't exist
    exp_dir = "./experiments/" + folder_name + "/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    return exp_dir


def share(data, name=None, borrow=True):
    """
    Transform data into Theano shared variable.
    """
    return theano.shared(np.asarray(data, dtype=theano.config.floatX), name=name, borrow=borrow)


def get_h5file_attribute(h5file, attr_key):
    try:
        attr_value = h5file.attrs[attr_key]
    except KeyError:
        raise Exception("Attribute {} is not present in {}".format(attr_key, h5file.filename))

    return attr_value


def get_h5file_data(h5file, data_key):
    try:
        data_value = h5file[data_key].value
    except KeyError:
        raise Exception("Data {} is not present in {}".format(data_key, h5file.filename))

    return data_value


def open_h5file(file_path):
    try:
        h5file = h5py.File(file_path, "r")
    except IOError:
        raise Exception("{} does not exist".format(file_path))

    return h5file


def distrib_balls_in_bins(n_balls, n_bins):
    """
    Uniformly distribute n_balls in n_bins
    """
    balls_per_bin = np.zeros((n_bins,), dtype=int)
    div, n_balls_remaining = divmod(n_balls, n_bins)
    balls_per_bin += div
    rand_idx = np.asarray(random.sample(range(n_bins), n_balls_remaining), dtype=int)
    balls_per_bin[rand_idx] += 1
    return balls_per_bin


def create_img_from_pred(vx, pred, shape):
    """
    Create a labelled image array from voxels and their predictions
    """
    pred_img = np.zeros(shape, dtype=np.uint8)
    if len(shape) == 2:
        pred_img[vx[:, 0], vx[:, 1]] = pred
    elif len(shape) == 3:
        pred_img[vx[:, 0], vx[:, 1], vx[:, 2]] = pred
    return pred_img


def analyse_classes(targets_scalar, verbose=True):
    """
    Compute various statistics about targets in a classification problem. The target of a point has to be an integer
    value.
    """
    a = np.bincount(targets_scalar)
    classes = np.nonzero(a)[0]
    n_classes = len(classes)
    if verbose:
        print("There are {} target classes in the dataset".format(n_classes))
    proportion_class = a[classes].astype(float, copy=False)
    proportion_class /= sum(proportion_class)

    if verbose:
        print("    The largest class represents {} % of the data".format(max(proportion_class) * 100))
        print("    The smallest class represents {} % of the data".format(min(proportion_class) * 100))

    return classes, proportion_class


def compute_dice(img_pred, img_true, n_classes_max):
    """
    Compute the DICE score between two segmentations
    """
    classes = np.unique(img_pred)
    if classes[0] == 0:
        classes = classes[1:]
    dices = np.zeros((n_classes_max,))

    for c in classes:
        class_pred = img_pred == c
        class_true = img_true == c
        class_common = class_true[class_pred]
        dices[c] = 2 * np.sum(np.asarray(class_common, dtype=float)) / (np.sum(class_pred) + np.sum(class_true))

    return dices


def compute_dice_symb(vec_pred, vec_true, n_classes_max):
    """
    Compute the DICE score between two segmentations in theano
    """
    vec_pred = vec_pred.dimshuffle((0, 'x'))
    vec_true = vec_true.dimshuffle((0, 'x'))

    classes = theano.shared(np.arange(1, n_classes_max))  # Start from 1 because we don't consider 0
    classes = classes.dimshuffle(('x', 0))

    binary_pred = T.cast(T.eq(vec_pred, classes), theano.config.floatX)
    binary_true = T.cast(T.eq(vec_true, classes), theano.config.floatX)
    binary_common = binary_pred * binary_true

    binary_pred_sum = T.sum(binary_pred, axis=0)
    binary_true_sum = T.sum(binary_true, axis=0)
    binary_common_sum = T.sum(binary_common, axis=0)
    no_zero = binary_common_sum.nonzero()

    return 2 * binary_common_sum[no_zero] / (binary_pred_sum[no_zero] + binary_true_sum[no_zero])


def compare_two_seg(pred_seg_path, true_seg_path):
    pred_seg = nib.load(pred_seg_path).get_data().squeeze()
    true_seg = nib.load(true_seg_path).get_data().squeeze()

    classes, true_volumes = analyse_classes(np.ravel(true_seg))
    dices = compute_dice(pred_seg, true_seg, len(classes)+1)
    dices = dices[1:]

    # Plot dice in function of log volume
    plt.plot(np.log10(true_volumes), dices, 'ro', label="one region")
    plt.xlabel('Log-volume of the region')
    plt.ylabel('Dice coefficient of the region')
    plt.savefig("./analysis/log_volume.png")

    # Plot dice in function of the sorted indices of the regions
    plt.figure()
    idx = np.argsort(dices)
    plt.plot(idx, dices[idx], 'ro', label="one region")
    plt.xlabel('Sorted indices of the regions (the higher the bigger the region)')
    plt.ylabel('Dice coefficient of the sorted region')
    plt.savefig("./analysis/dices_sorted.png")


def error_rate_symb(y1, y2):
    """Return the symbolic error rate
    Args:
        y1, y2 (theano.tensor.TensorType): 2D arrays to compare. Each row represents a point and each column a class.
    """
    return T.mean(T.neq(T.argmax(y1, axis=1), T.argmax(y2, axis=1)))


def error_rate(y1, y2):
    """Return the numpy error rate
    Args:
        y1, y2 (numpy 2D array): 2D arrays to compare. Each row represents a point and each column a class.
    """
    return np.mean(np.argmax(y1, axis=1) != np.argmax(y2, axis=1))