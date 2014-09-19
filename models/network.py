__author__ = 'adeb'

import sys
import h5py

from spynet.utils.utilities import get_h5file_attribute, error_rate
from spynet.models.layer import *
from spynet.models.layer_block import *
from spynet.models import neuron_type


class Network(object):
    """
    Abstract class whose child classes define custom user networks.

    Attributes:
        n_in (int): number of inputs of the network
        n_out (int): Number of outputs of the network

        ls_layers (list): list of the layers composing the network
        ls_params (list): list of arrays of parameters of all the layers
    """
    def __init__(self):
        self.name = self.__class__.__name__

        self.n_in = None
        self.n_out = None

        self.ls_layers = []
        self.ls_params = []

    def init_common(self, n_in, n_out):
        print "Initialize the model ..."
        self.n_in = n_in
        self.n_out = n_out

    def concatenate_parameters(self):
        self.ls_params = []
        for l in self.ls_layers:
            self.ls_params += l.params

    def forward(self, in_batch, batch_size):
        """Return the output of the network
        Args:
            in_batch (theano.tensor.TensorType): input batch of the network
        Returns:
            (theano.tensor.TensorType): outputs of the network
        """
        out_batch = [in_batch]
        for l in self.ls_layers:
            out_batch = l.forward(out_batch, batch_size)

        return out_batch[0]

    def generate_testing_function(self, batch_size):
        """
        Generate a C-compiled function that can be used to compute the output of the network from an input batch
        Args:
            batch_size (int): the input of the returned function will be a batch of batch_size elements
        Returns:
            (function): function that returns the output of the network for a given input batch
        """
        in_batch = T.matrix('in_batch')  # Minibatch input matrix
        y_pred = self.forward(in_batch, batch_size)  # Output of the network
        return theano.function([in_batch], y_pred)

    def predict(self, in_numpy_array, batch_size_limit):
        """
        User-friendly function to return the outputs of provided inputs without worrying about batch_size.
        Args:
            in_numpy_array (2D array): dataset in which rows are datapoints
            batch_size_limit (int): limit size of a batch (should be what the GPU memory can support (or the RAM))
        Returns:
            pred (2D array): outputs of the network for the given inputs
        """
        n_inputs = in_numpy_array.shape[0]
        out_pred = np.zeros((n_inputs, self.n_out), dtype=np.float32)  # Will store the output predictions
        batch_size = min(batch_size_limit, n_inputs)
        pred_fun = self.generate_testing_function(batch_size)

        n_batches, n_rest = divmod(n_inputs, batch_size)

        print "--------------------"
        for b in xrange(n_batches):
            sys.stdout.write("\r        Prediction: {}%".format(100*b/n_batches))
            sys.stdout.flush()
            id0 = b*batch_size
            id1 = id0 + batch_size
            out_pred[id0:id1] = pred_fun(in_numpy_array[id0:id1])

        if n_rest > 0:
            pred_fun_res = self.generate_testing_function(n_rest)
            out_pred[n_batches*batch_size:] = pred_fun_res(in_numpy_array[n_batches*batch_size:])

        return out_pred

    def predict_from_generator(self, batches_generator, scaler, pred_functions=None):
        """
        Returns the predictions of the batches of voxels, features and targets yielded by the batches_generator
        """
        if pred_functions is None:
            pred_functions = {}
        ls_vx = []
        ls_pred = []
        id_batch = 0
        for vx_batch, patch_batch, tg_batch in batches_generator:
            id_batch += 1

            batch_size_current = len(vx_batch)
            if batch_size_current not in pred_functions:
                pred_functions[batch_size_current] = self.generate_testing_function(batch_size_current)

            if scaler is not None:
                scaler.scale(patch_batch)

            pred_raw = pred_functions[batch_size_current](patch_batch)

            pred = np.argmax(pred_raw, axis=1)
            err = error_rate(pred, np.argmax(tg_batch, axis=1))
            print "     {}".format(err)

            ls_vx.append(vx_batch)
            ls_pred.append(pred)

        # Count the number of voxels
        n_vx = 0
        for vx in ls_vx:
            n_vx += vx.shape[0]

        # Aggregate the data
        vx_all = np.zeros((n_vx, 3), dtype=int)
        pred_all = np.zeros((n_vx,), dtype=int)
        idx = 0
        for vx, pred in zip(ls_vx, ls_pred):
            next_idx = idx+vx.shape[0]
            vx_all[idx:next_idx] = vx
            pred_all[idx:next_idx] = pred
            idx = next_idx

        return vx_all, pred_all

    def save_parameters(self, file_path):
        """
        Save parameters (weights, biases, scaling info) of the network in an hdf5 file
        """
        f = h5py.File(file_path, "w")
        f.attrs['network_type'] = self.__class__.__name__
        f.attrs['n_in'] = self.n_in
        f.attrs['n_out'] = self.n_out
        self.save_parameters_virtual(f)
        for i, l in enumerate(self.ls_layers):
            l.save_parameters(f, "layer" + str(i))
        f.close()

    def save_parameters_virtual(self, h5file):
        raise NotImplementedError

    def load_parameters(self, h5file):
        """
        Load parameters (weights, biases, scaling info) of the network from an hdf5 file
        If reset_network is True, then the layers are re-created.
        If reset_network is False, then the layers are only updated
        """
        self.n_in = int(get_h5file_attribute(h5file, "n_in"))
        self.n_out = int(get_h5file_attribute(h5file, "n_out"))
        self.load_parameters_virtual(h5file)
        for i, l in enumerate(self.ls_layers):
            l.load_parameters(h5file, "layer" + str(i))

    def load_parameters_virtual(self, h5file):
        raise NotImplementedError

    def __str__(self):
        n_parameters = 0
        for p in self.ls_params:
            n_parameters += p.get_value().size
        msg = "This network has the following layers: \n"
        for i, l in enumerate(self.ls_layers):
            msg += "------- Layer {} ------- \n".format(i)
            msg += l.__str__()
        msg += "The type of this network is {}. It has {} inputs, {} outputs and {} parameters \n"\
            .format(self.name, self.n_in, self.n_out, n_parameters)
        return msg

    def export_params(self):
        """
        Return the real value of Theano shared variables params.
        """
        params = []
        for p in self.ls_params:
            params.append(p.get_value())
        return params

    def import_params(self, params):
        """
        Update Theano shared variable self.params with numpy variable params.
        """
        for p, p_sym in zip(params, self.ls_params):
            p_sym.set_value(p)

    def get_layer(self, idx_layer):
        return self.ls_layers[idx_layer]

    def update_params(self):
        self.ls_params = []
        for l in self.ls_layers:
            l.update_params()
            self.ls_params += l.params

        # Check that there no duplicates (if layers share weights) (TODO: should be put in a set since the beginning)
        self.ls_params = list(set(self.ls_params))


class MLP(Network):
    """
    Multi-layer perceptron
    """
    def __init__(self):
        Network.__init__(self)

    def init(self, ls_layer_size, neuron_function=neuron_type.NeuronRELU()):
        Network.init_common(self, ls_layer_size[0], ls_layer_size[-1])

        ls_block = []
        for i in xrange(len(ls_layer_size)-2):
            ls_block.append(LayerBlockFullyConnected(neuron_function, ls_layer_size[i], ls_layer_size[i+1]))

        # Last layer is softmax
        ls_block.append(LayerBlockFullyConnected(neuron_type.NeuronSoftmax(), ls_layer_size[-2], ls_layer_size[-1]))

        self.ls_layers = convert_blocks_into_feed_forward_layers(ls_block)

        self.concatenate_parameters()

    def save_parameters_virtual(self, h5file):
        pass

    def load_parameters_virtual(self, h5file):
        # This function is not working properly, see issue 2 on GitHub
        pass


class ConvNet2DExample(Network):
    """
    2D convnet
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


class ConvNet3DExample(Network):
    """
    3D convnet
    """
    def __init__(self):
        Network.__init__(self)
        self.in_height = None
        self.in_width = None
        self.in_depth = None

    def init(self, patch_height, patch_width, patch_depth, n_out):
        Network.init_common(self, patch_height*patch_width*patch_depth, n_out)

        self.in_height = patch_height
        self.in_width = patch_width
        self.in_depth = patch_depth

        neuron_relu = neuron_type.NeuronRELU()

        # Layer 0
        filter_map_0_shape = np.array([patch_height, patch_width, patch_depth], dtype=int)
        filter_0_shape = np.array([2, 2, 2], dtype=int)
        pool_0_shape = np.array([2, 2, 2], dtype=int)
        n_kern0 = 20
        block0 = LayerBlockConvPool3D(neuron_relu,
                                      1, tuple(filter_map_0_shape),
                                      n_kern0, tuple(filter_0_shape),
                                      poolsize=tuple(pool_0_shape))

        # Layer 1
        filter_map_1_shape = (filter_map_0_shape - filter_0_shape + 1) / pool_0_shape
        filter_1_shape = np.array([2, 2, 2], dtype=int)
        pool_1_shape = np.array([2, 2, 2], dtype=int)
        n_kern1 = 50
        block1 = LayerBlockConvPool3D(neuron_relu,
                                      n_kern0, tuple(filter_map_1_shape),
                                      n_kern1, tuple(filter_1_shape),
                                      poolsize=tuple(pool_1_shape))

        # Layer 2
        filter_map_2_shape = (filter_map_1_shape - filter_1_shape + 1) / pool_1_shape
        n_in2 = n_kern1 * np.prod(filter_map_2_shape)
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
        h5file.attrs['in_depth'] = self.in_depth

    def load_parameters_virtual(self, h5file):
        self.in_height = int(h5file.attrs["in_height"])
        self.in_width = int(h5file.attrs["in_width"])
        self.in_depth = int(h5file.attrs["in_depth"])
