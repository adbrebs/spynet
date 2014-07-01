__author__ = 'adeb'

import theano.tensor as T


class Layer():
    """
    This abstract class represents a layer of a neural network. A spynet layer is more general than common definition
    of a lyer of neurons in the sense that a spynet layer is not necessary composed of neurons. As described in the
    child classes, a spynet layer can merge or divide the inputs.
    """
    def __init__(self):
        self.params = []

    def forward(self, ls_input, batch_size):
        """Return the output of the layer block
        Args:
            ls_input (list of theano.tensor.TensorType): input of the layer
        Returns:
            (list of theano.tensor.TensorType): output of the layer
        """
        raise NotImplementedError

    def save_parameters(self, h5file, name):
        """
        Save all parameters of the block layer in a hdf5 file.
        """
        pass

    def load_parameters(self, h5file, name):
        """
        Load all parameters of the block layer in a hdf5 file.
        """
        pass

    def __str__(self):
        """
        Should end with \n.
        """
        raise NotImplementedError


class LayerMergeFeatures(Layer):
    """
    Merge the output features of the previous layer.
    """
    def __init__(self):
        Layer.__init__(self)

    def forward(self, ls_inputs, batch_size):
        return [T.concatenate(ls_inputs, axis=1)]

    def __str__(self):
        return "Merging layer\n"


class LayerDivideFeatures(Layer):
    """
    Divide the output features of the previous layer so that different blocks can be used in the next layer.
    Attributes:
        ls_split_idx: List of indices of where the input features should be divided. For example, if
            ls_split_idx = [0 700 3000], the features will be divided in two: the first [0 700] features on one
            side and the other [700 3000] features on the other side.
    """
    def __init__(self, ls_split_idx):
        Layer.__init__(self)
        self.ls_split_idx = ls_split_idx

    def forward(self, ls_inputs, batch_size):
        if len(ls_inputs) != 1:
            raise Exception("LayerDivide's input should be of length 1")
        input = ls_inputs[0]

        ls_outputs = []
        for i in xrange(len(self.ls_split_idx) - 1):
            s = slice(self.ls_split_idx[i], self.ls_split_idx[i+1])
            ls_outputs.append(input[:, s])
        return ls_outputs

    def __str__(self):
        return "Dividing layer\n"


class LayerOfBlocks(Layer):
    """
    Layer composed of blocks of neurons. A LayerOfBlocks has the same meaning as a layer in the neural network
    vocabulary.
    Attributes:
        ls_layer_blocks: List of LayerBlock objects
    """
    def __init__(self, ls_layer_blocks):
        Layer.__init__(self)
        self.ls_layer_blocks = ls_layer_blocks

        self.params = []
        for l in self.ls_layer_blocks:
            self.params += l.params

    def forward(self, ls_inputs, batch_size):
        ls_outputs = []
        for x, layer_block in zip(ls_inputs, self.ls_layer_blocks):
            ls_outputs.append(layer_block.forward(x, batch_size))
        return ls_outputs

    def save_parameters(self, h5file, name):
        for i, l in enumerate(self.ls_layer_blocks):
            l.save_parameters(h5file, name + "/block" + str(i))

    def load_parameters(self, h5file, name):
        for i, l in enumerate(self.ls_layer_blocks):
            l.load_parameters(h5file, name + "/block" + str(i))

    def __str__(self):
        msg = "Layer composed of the following block(s):\n"
        for i, l in enumerate(self.ls_layer_blocks):
            msg += "Block " + str(i) + ":\n" + l.__str__() + "\n"
        return msg


def convert_blocks_into_feed_forward_layers(ls_layer_blocks):
    """
    Convenient function to convert a list of layer blocks into a list of LayerOfBlocks, each LayerOfBlock containing a
    single block. It is useful when you don't need to divided the features of your data.
    """
    ls_layers = []
    for layer_block in ls_layer_blocks:
        ls_layers.append(LayerOfBlocks([layer_block]))
    return ls_layers