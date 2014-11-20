__author__ = 'adeb'


import numpy as np

import theano
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv, conv3d2d
from theano.tensor.shared_randomstreams import RandomStreams

from spynet.utils.utilities import share, get_h5file_data
from spynet.models.max_pool_3d import max_pool_3d


class LayerBlock():
    """
    Abstract class that represents a function from an input space to an output space.
    It is the building block of a Layer object.
    """
    name = None

    def __init__(self):
        self.params = []

    def forward(self, x, batch_size, run_time):
        """Return the output of the layer block
        Args:
            x (theano.tensor.TensorType): input of the layer block
            batch_size (int): size of the batch of data being processed by the network
            run_time (boolean): equals true when the function is used at runtime and false when it is used during
                training. This is useful for dropout.
        Returns:
            (theano.tensor.TensorType): output of the layer block
        """
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        """
        Save all parameters of the layer block in a hdf5 file.
        """
        pass

    def load_parameters(self, h5file, name):
        """
        Load all parameters of the layer block in a hdf5 file.
        """
        pass

    def update_params(self):
        pass

    def __str__(self):
        msg = "[{}] \n".format(self.name)
        return msg


class LayerBlockIdentity(LayerBlock):
    """
    Identity function
    """
    name = "Identity Layer block"

    def __init__(self):
        LayerBlock.__init__(self)

    def forward(self, x, batch_size, run_time):
        return x


class LayerBlockNoise(LayerBlock):
    """
    Noise layer block that adds a random signal on the fly
    """
    def __init__(self):
        LayerBlock.__init__(self)
        numpy_rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(numpy_rng.randint(2**30))


class LayerBlockNoiseDropoutBernoulli(LayerBlockNoise):
    """
    Noise block layer that adds bernoulli noise on the fly
    """
    name = "Bernoulli Layer block"

    def __init__(self, bernoulli_p):
        LayerBlockNoise.__init__(self)
        self.bernoulli_p = bernoulli_p

    def forward(self, x, batch_size, run_time):
        if run_time:
            return x * self.bernoulli_p
        else:
            return x * self.theano_rng.binomial(size=x.shape, n=1, p=self.bernoulli_p, dtype=theano.config.floatX)


class LayerBlockGaussianNoise(LayerBlockNoise):
    """
    Noise block layer that adds gaussian noise on the fly
    """
    name = "Gaussian noise Layer block"

    def __init__(self):
        LayerBlockNoise.__init__(self)

    def forward(self, x, batch_size, run_time):
        return x + self.theano_rng.normal(size=x.shape, avg=0, std=0.2, dtype=theano.config.floatX)


class LayerBlockMultiplication(LayerBlock):
    """
    Block that multiplies the input elementwise by a vector of the same size
    """
    name = "Multiplication Layer block"

    def __init__(self, vec):
        LayerBlock.__init__(self)
        self.vec = share(vec)

    def forward(self, x, batch_size, run_time):
        return x * self.vec


class LayerBlockNormalization(LayerBlock):
    """
    Block that normalizes the input so it sums to one
    """
    name = "Normalization Layer block"

    def __init__(self):
        LayerBlock.__init__(self)

    def forward(self, x, batch_size, run_time):
        return x / theano.tensor.sum(x)


class LayerBlockOfNeurons(LayerBlock):
    """
    Abstract class defining a group of neurons.

    Attributes:
        name (string): Name of the layer block (used for printing or writing)
        w (theano shared numpy array): Weights of the layer block
        b (theano shared numpy array): Biases of the layer block
        params (list): [w,b]
        neuron_type (NeuronType object): defines the type of the neurons of the layer block
    """
    def __init__(self, neuron_type):
        LayerBlock.__init__(self)
        self.w = None
        self.b = None
        self.neuron_type = neuron_type

    def init_parameters(self, w_shape, b_shape):
        w_bound = self.compute_bound_parameters_virtual()

        # initialize weights with random weights
        self.w = share(np.asarray(
            np.random.uniform(low=-w_bound, high=w_bound, size=w_shape),
            dtype=theano.config.floatX), "w")

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = 0.1 + np.zeros(b_shape, dtype=theano.config.floatX)  # Slightly positive for RELU units
        self.b = share(b_values, "b")

        self.update_params()

    def compute_bound_parameters_virtual(self):
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        h5file.create_dataset(name + "/w", data=self.w.get_value(), dtype='f')
        h5file.create_dataset(name + "/b", data=self.b.get_value(), dtype='f')

    def load_parameters(self, h5file, name):
        self.w.set_value(get_h5file_data(h5file, name + "/w"), borrow=True)
        self.b.set_value(get_h5file_data(h5file, name + "/b"), borrow=True)

    def update_params(self):
        self.params = [self.w, self.b]

    def __str__(self):
        msg = "[{}] with [{}] \n".format(self.name, self.neuron_type)
        msg += self.print_virtual()
        n_parameters = 0
        for p in self.params:
            n_parameters += p.get_value().size
        msg += "Number of parameters: {} \n".format(n_parameters)
        return msg

    def print_virtual(self):
        return ""


class LayerBlockFullyConnected(LayerBlockOfNeurons):
    """
    Layer block in which each input is connected to all the block neurons
    """
    name = "Fully connected layer block"

    def __init__(self, neuron_type, n_in, n_out):
        LayerBlockOfNeurons.__init__(self, neuron_type)

        self.n_in = n_in
        self.n_out = n_out

        self.init_parameters((self.n_in, self.n_out), (self.n_out,))

    def compute_bound_parameters_virtual(self):
        return np.sqrt(6. / (self.n_in + self.n_out))

    def set_w(self, new_w):
        self.w.set_value(new_w, borrow=True)
        self.n_in, self.n_out = new_w.shape

    def forward(self, x, batch_size, run_time):
        return self.neuron_type.activation_function(theano.tensor.dot(x, self.w) + self.b)

    def print_virtual(self):
        return "Number of inputs: {} \nNumber of outputs: {}\n".format(self.n_in, self.n_out)


class LayerBlockConv2DAbstract(LayerBlockOfNeurons):
    """
    Abstract class defining common components of LayerConv2D and LayerConvPool2D
    """
    def __init__(self, neuron_type, in_shape, flt_shape):
        """
        Args:
            in_shape (tuple or list of length 3):
            (num input feature maps, image height, image width)

            flt_shape (tuple or list of length 4):
            (number of filters, num input feature maps, filter height, filter width)
        """
        LayerBlockOfNeurons.__init__(self, neuron_type)

        self.in_shape = in_shape
        self.filter_shape = flt_shape

        if in_shape[0] != flt_shape[1]:
            raise Exception("The number of feature maps is not consistent")

        self.init_parameters(flt_shape, (flt_shape[0],))

    def forward(self, x, batch_size, run_time):
        img_batch_shape = (batch_size,) + self.in_shape

        x = x.reshape(img_batch_shape)

        # Convolve input feature maps with filters
        conv_out = conv.conv2d(input=x,
                               filters=self.w,
                               image_shape=img_batch_shape,
                               filter_shape=self.filter_shape)

        return self.forward_virtual(conv_out)

    def forward_virtual(self, conv_out):
        raise NotImplementedError

    def print_virtual(self):
        return "Image shape: {}\nFilter shape: {}\n".format(self.in_shape, self.filter_shape)


class LayerBlockConv2D(LayerBlockConv2DAbstract):
    """
    2D convolutional layer block
    """
    name = "2D convolutional layer block"

    def __init__(self, neuron_type, in_shape, flt_shape):
        LayerBlockConv2DAbstract.__init__(self, neuron_type, in_shape, flt_shape)

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = self.filter_shape[0] * np.prod(self.filter_shape[2:])

        return np.sqrt(6. / (fan_in + fan_out))

    def forward_virtual(self, conv_out):
        return self.neuron_type.activation_function(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)


class LayerBlockConvPool2D(LayerBlockConv2DAbstract):
    """
    2D convolutional layer + pooling layer. The reason for not having a separate pooling layer is that the combination
    of the two layer blocks can be optimized.
    """
    name = "2D convolutional + pooling layer"

    def __init__(self, neuron_type, in_shape, flt_shape, poolsize=(2, 2)):
        self.poolsize = poolsize
        LayerBlockConv2DAbstract.__init__(self, neuron_type, in_shape, flt_shape)

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.poolsize))

        return np.sqrt(6. / (fan_in + fan_out))

    def forward_virtual(self, conv_out):
        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=self.poolsize,
                                            ignore_border=True)

        return self.neuron_type.activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')).flatten(2)

    def print_virtual(self):
        return LayerBlockConv2DAbstract.print_virtual(self) + "Pool size: {}\n".format(self.poolsize)


class LayerBlockConvPool3D(LayerBlockOfNeurons):
    """
    3D convolutional layer block + pooling layer block
    """
    name = "3D convolutional + pooling layer block"

    def __init__(self, neuron_type, in_channels, in_shape, flt_channels, flt_shape, poolsize):
        """
        Args:
            in_channels (int): number of input channels
            in_shape (tuple of length 3): shape of the input (in_width, in_height, in_depth)

            flt_channels (int):
            flt_shape (tuple of length 3): shape of the filters (flt_depth, flt_height, flt_width)

            poolsize (tuple of length 3): window of the pooling operation
        """
        LayerBlockOfNeurons.__init__(self, neuron_type)

        in_width, in_height, in_depth = self.in_shape = in_shape
        flt_width, flt_height, flt_depth = self.flt_shape = flt_shape
        self.in_channels = in_channels
        self.flt_channels = flt_channels

        self.image_shape = (in_depth, in_channels, in_height, in_width)
        self.filter_shape = (flt_channels, flt_depth, in_channels, flt_height, flt_width)
        self.poolsize = poolsize

        self.init_parameters(self.filter_shape, (self.filter_shape[0],))

    def compute_bound_parameters_virtual(self):
        fan_in = np.prod(self.in_shape)
        fan_out = self.flt_channels * np.prod(self.flt_shape) / np.prod(self.poolsize)

        return np.sqrt(6. / (fan_in + fan_out))

    def forward(self, x, batch_size, run_time):
        img_batch_shape = (batch_size,) + self.image_shape

        x = x.reshape(img_batch_shape)

        # Convolve input feature maps with filters
        conv_out = conv3d2d.conv3d(signals=x,
                                   filters=self.w,
                                   signals_shape=img_batch_shape,
                                   filters_shape=self.filter_shape,
                                   border_mode='valid')

        perm = [0, 2, 1, 3, 4]  # Permutation is needed due to the pooling function prototype
        pooled_out = max_pool_3d(conv_out.dimshuffle(perm), self.poolsize, ignore_border=True)

        return self.neuron_type.activation_function(pooled_out.dimshuffle(perm)
                                                    + self.b.dimshuffle('x', 'x', 0, 'x', 'x')).flatten(2)

    def print_virtual(self):
        return "Image shape: {} \n Filter shape: {} \n Pool size: {} \n".format(
            self.image_shape, self.filter_shape, self.poolsize)