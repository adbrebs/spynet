__author__ = 'adeb'

import os
import sys
from shutil import copy2
import inspect
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
        cf_path = default_file
    else:
        cf_path = str(sys.argv[1])

    cf = __import__(os.path.splitext(cf_path)[0])
    data_path = cf.data_path

    # Create the folder if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Copy the config file
    copy2(cf_path, data_path)

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


def analyse_classes(targets_scalar, title="", verbose=True):
    """
    Compute various statistics about targets in a classification problem. The target of a point has to be an integer
    value.
    """
    a = np.bincount(targets_scalar)
    classes = np.nonzero(a)[0]
    n_classes = len(classes)
    if verbose:
        print title
        print("     There are {} datapoints in the dataset".format(targets_scalar.shape[0]))
        print("     There are {} target classes in the dataset".format(n_classes))
    proportion_class = a[classes].astype(float, copy=False)
    proportion_class /= sum(proportion_class)

    idx_max = np.argmax(proportion_class)
    idx_min = np.argmin(proportion_class)

    if verbose:
        print("     The largest class is {} and represents {} % of the data"
              .format(idx_max, proportion_class[idx_max] * 100))
        print("    The smallest class is {} and represents {} % of the data"
              .format(idx_min, proportion_class[idx_min] * 100))

    return classes, proportion_class


def compute_dice(pred, true, n_classes):
    """
    Compute the DICE score between two vectors pred and true
    """
    counts = count_common_classes(pred, true, n_classes)
    dices = 2 * np.asarray(counts[:, 2], dtype=float) / (counts[:, 0] + counts[:, 1])

    return dices[~np.isnan(dices)]


def compute_dice_from_counts(counts):
    """
    Compute the DICE score from function count_common_classes
    """
    dices = 2 * np.asarray(counts[:, 2], dtype=float) / (counts[:, 0] + counts[:, 1])
    return dices[~np.isnan(dices)]


def count_common_classes(pred, true, n_classes):
    """
    pred and true are one-dimensional vectors of integers
    For each class c, compute three values:
        - the number of elements of classes c in pred
        - the number of elements of classes c in true
        - the number of common elements of classes c in pred and true
    """
    counts = np.zeros((n_classes,3))

    for c in xrange(1, n_classes):  # Start from 1 because we don't consider 0
        class_pred = pred == c
        class_true = true == c

        counts[c, 0] = np.sum(class_pred)
        counts[c, 1] = np.sum(class_true)
        counts[c, 2] = np.sum(class_true * class_pred)

    return counts[1:, :]


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
        y1, y2 (theano.tensor.TensorType): 1D arrays to compare. Each element represents a point.
    """
    return T.mean(T.neq(y1, y2))


def error_rate_matrix_symb(y1, y2):
    """Return the symbolic error rate
    Args:
        y1, y2 (theano.tensor.TensorType): 2D arrays to compare. Each row represents a point and each column a class.
    """
    return T.mean(T.neq(T.argmax(y1, axis=1), T.argmax(y2, axis=1)))


def error_rate(y1, y2):
    """Return the numpy error rate
    Args:
        y1, y2 (numpy 2D vectors): 1D arrays to compare. Each element represents a point.
    """
    return np.mean(y1 != y2)


def error_rate_matrix(y1, y2):
    """Return the numpy error rate
    Args:
        y1, y2 (numpy 2D array): 2D arrays to compare. Each row represents a point and each column a class.
    """
    return np.mean(np.argmax(y1, axis=1) != np.argmax(y2, axis=1))


def MSE(y1, y2):
    return np.mean(np.sum((y1 - y2) * (y1 - y2), axis=1))


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    original function from deeplearning.net
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array