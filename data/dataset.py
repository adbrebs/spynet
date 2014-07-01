__author__ = 'adeb'


from datetime import datetime
import h5py
import numpy as np

from spynet.utils.utilities import open_h5file, share


class DatasetWithoutOutputs(object):
    """
    Class to store a numpy 2D array called inputs that represent datapoints as rows and features as columns. Each
    datapoint does not have any corresponding stored output (see child class to include them).
    This class can also load the data on the GPU by transforming it into theano shared variables.
    You can load/save from/in hdf5 files.

    You can inherit from this class if you want to specify your own data attributes. If so, you may want to overwrite
    these methods: shuffle_data_virtual, write_virtual, read_virtual

    Attributes:
        inputs (2D numpy array): rows represent datapoints and columns represent features
        n_in_features (int): number of input featurse
        n_data (int): number of datapoints
        is_perm (boolean): indicates if the dataset is shuffled or not

    """
    def __init__(self):

        self._inputs = None
        self._inputs_shared = None
        self.n_data = None
        self.n_in_features = None

        self.is_perm = False

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value
        self.n_data, self.n_in_features = value.shape

    @property
    def inputs_shared(self):
        self._inputs_shared = share(self.inputs)
        return self._inputs_shared

    @inputs_shared.setter
    def inputs_shared(self, value):
        raise Exception("The shared variable should not be replaced.")

    def shuffle_data(self):
        """
        Shuffle the dataset.
        """
        perm = np.random.permutation(self.n_data)
        self.inputs = self.inputs[perm]
        self.shuffle_data_virtual(perm)

    def shuffle_data_virtual(self, perm):
        """
        Should be overwritten if you define a child class with attributes that need to be shuffled as well.
        """
        pass

    def write(self, file_path):
        """
        write the dataset in a hdf5 file.
        """
        h5file = h5py.File(file_path, "w")
        h5file.create_dataset("inputs", data=self.inputs, dtype='f')

        h5file.attrs['creation_date'] = str(datetime.now())
        h5file.attrs['n_data'] = self.n_data
        h5file.attrs['n_in_features'] = self.n_in_features
        h5file.attrs['is_perm'] = self.is_perm

        self.write_virtual(h5file)

        h5file.close()

    def write_virtual(self, h5file):
        pass

    def read(self, file_path):
        """
        load the dataset from a hdf5 file.
        """
        h5file = open_h5file(file_path)
        self.inputs = h5file["inputs"].value
        self.is_perm = bool(h5file.attrs['is_perm'])
        self.read_virtual(h5file)
        h5file.close()

    def read_virtual(self, h5file):
        pass

    def create_sub_dataset(self, slice_idx):
        """
        Create a sub dataset from a slice of indices slice_idx
        """
        ds = DatasetWithoutOutputs()
        ds.inputs = self.inputs[slice_idx]
        return ds


class DatasetWithOutputs(DatasetWithoutOutputs):
    """
    Class to store two numpy 2D arrays respectively called inputs and outputs. Rows represent the datapoints. Column of
    inputs are the input features. Columns of outputs represent output variables.
    This class can also load the data on the GPU by transforming it into theano shared variables.
    You can load/save from/in hdf5 files.

    You can inherit from this class if you want to specify your own data attributes. If so, you may want to overwrite
    these methods: shuffle_data_virtual, write_virtual, read_virtual

    Attributes:
        outputs (2D numpy array): if known, corresponding outputs of the inputs
        n_out_features (int): number of output features

    """
    def __init__(self):
        DatasetWithoutOutputs.__init__(self)

        self._outputs = None
        self._outputs_shared = None
        self.n_out_features = None

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value
        self.n_out_features = value.shape[1]

    @property
    def outputs_shared(self):
        self._outputs_shared = share(self.outputs)
        return self._outputs_shared

    @outputs_shared.setter
    def outputs_shared(self, value):
        raise Exception("The shared variable should not be replaced.")

    def shuffle_data_virtual(self, perm):
        self.outputs = self.outputs[perm]

    def share_data_virtual(self):
        self.outputs_shared = share(self.outputs)

    def write_virtual(self, h5file):
        h5file.create_dataset("outputs", data=self.outputs, dtype='f')
        h5file.attrs['n_out_features'] = self.n_out_features

    def read_virtual(self, h5file):
        self.outputs = h5file["outputs"].value

    def create_sub_dataset(self, slice_idx):
        """
        Create a sub dataset from a slice of indices slice_idx
        """
        ds = DatasetWithOutputs()
        ds.inputs = self.inputs[slice_idx]
        ds.outputs = self.outputs[slice_idx]
        return ds


def split_dataset(ds, proportion):
    """
    Split a dataset in two sub datasets according to proportion.
    Delete the original dataset.
    """
    if proportion < 0 or proportion > 1:
        raise Exception("the proportion to split the dataset should be between 0 and 1")

    split = int(proportion * ds.n_data)

    ds1 = ds.create_sub_dataset(slice(0, split))
    ds2 = ds.create_sub_dataset(slice(split, None))

    del ds

    return ds1, ds2
