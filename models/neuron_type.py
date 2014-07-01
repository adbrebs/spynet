__author__ = 'adeb'

from theano import tensor as T


class NeuronType():
    """
    Abstract class defining a neuron type. This class defines the activation function of the neuron.
    """
    name = None

    def __init__(self):
        pass

    def activation_function(self, x):
        raise NotImplementedError

    def __str__(self):
        return "Neuron type: {}".format(self.name)


class NeuronTanh(NeuronType):
    name = "Tanh"

    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.tanh(x)


class NeuronSoftmax(NeuronType):
    name = "SoftMax"

    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.nnet.softmax(x)


class NeuronRELU(NeuronType):
    """
    Rectified linear unit
    """
    name = "RELU"

    def __init__(self):
        NeuronType.__init__(self)

    def activation_function(self, x):
        return T.switch(x > 0., x, 0)
