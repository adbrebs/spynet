__author__ = 'adeb'

from spynet.models.layer_block import *
import spynet.models.neuron_type as neuron_type
from spynet.models.layer import *
from spynet.models.network import Network


class NetworkMNIST(Network):
    """
    2D convnet for MNIST dataset
    """
    def __init__(self):
        Network.__init__(self)
        self.in_width = None
        self.in_height = None

    def init(self, patch_height, patch_width, n_out):
        Network.init_common(self, patch_height*patch_width, n_out)

        self.in_height = patch_height
        self.in_width = patch_width

        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        kernel_height0 = 5
        kernel_width0 = 5
        pool_size_height0 = 2
        pool_size_width0 = 2
        n_kern0 = 20
        block0 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(1, patch_height, patch_width),
                                      flt_shape=(n_kern0, 1, kernel_height0, kernel_width0),
                                      poolsize=(pool_size_height0, pool_size_width0))

        # Layer 1
        filter_map_height1 = (patch_height - kernel_height0 + 1) / pool_size_height0
        filter_map_width1 = (patch_width - kernel_width0 + 1) / pool_size_width0
        kernel_height1 = 5
        kernel_width1 = 5
        pool_size_height1 = 2
        pool_size_width1 = 2
        n_kern1 = 50
        block1 = LayerBlockConvPool2D(neuron_relu,
                                      in_shape=(n_kern0, filter_map_height1, filter_map_width1),
                                      flt_shape=(n_kern1, n_kern0, kernel_height1, kernel_width1),
                                      poolsize=(pool_size_height1, pool_size_width1))

        # Layer 2
        filter_map_height2 = (filter_map_height1 - kernel_height1 + 1) / pool_size_height1
        filter_map_with2 = (filter_map_width1 - kernel_width1 + 1) / pool_size_width1
        n_in2 = n_kern1 * filter_map_height2 * filter_map_with2
        n_out2 = 500
        block2 = LayerBlockFullyConnected(neuron_relu, n_in=n_in2, n_out=n_out2)

        # Layer 3
        block3 = LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), n_in=n_out2, n_out=self.n_out)

        self.ls_layers = convert_blocks_into_feed_forward_layers([block0, block1, block2, block3])

        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def save_parameters_virtual(self, h5file):
        h5file.attrs['in_height'] = self.in_height
        h5file.attrs['in_width'] = self.in_width

    def load_parameters_virtual(self, h5file):
        self.in_height = int(h5file.attrs["in_height"])
        self.in_width = int(h5file.attrs["in_width"])
        self.init(self.in_height, self.in_width, self.n_out)