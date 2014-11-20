__author__ = 'adeb'

# Hack to be able to run this module
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import matplotlib
matplotlib.use('Agg')
import PIL

from spynet.models.network import *
from spynet.models.neuron_type import *
from spynet.data.dataset import Dataset
from spynet.utils.utilities import open_h5file, tile_raster_images, MSE


if __name__ == '__main__':

    mode = "drop"

    experiment_path = "./experiments/mnist_example/"
    data_path = "./datasets/mnist/"

    testing_data_path = data_path + "test.h5"
    ds_testing = Dataset.create_and_read(testing_data_path)

    # Load the network
    net = AutoEncoder()
    net.init([28**2, 256, 28**2], dropout=True, dropout_p=[0.5], neuron_function=NeuronSigmoid())
    net.load_parameters(open_h5file(experiment_path + "netdrop.net"))

    i = ds_testing.inputs[0:10,:]
    e = net.predict(i, 10)

    print ""
    print MSE(e,i)

    image = PIL.Image.fromarray(tile_raster_images(X=net.ls_layers[0].ls_layer_blocks[0].w.get_value(borrow=True).T,
                 img_shape=(28, 28), tile_shape=(16, 16),
                 tile_spacing=(1, 1)))
    image.save(experiment_path + "filters" + mode + ".png")

    image = PIL.Image.fromarray(tile_raster_images(X=i,
                                                   img_shape=(28, 28), tile_shape=(1, 10),
                                                   tile_spacing=(1, 1)))
    image.save(experiment_path + "i" + mode + ".png")

    image = PIL.Image.fromarray(tile_raster_images(X=e,
                                                   img_shape=(28, 28), tile_shape=(1, 10),
                                                   tile_spacing=(1, 1)))
    image.save(experiment_path + "e" + mode + ".png")