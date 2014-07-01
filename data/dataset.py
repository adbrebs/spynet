__author__ = 'adeb'


from datetime import datetime
import h5py
import numpy as np

from spynet.utils.utilities import open_h5file, share


class Dataset(object):
    """
    Class to store a dataset composed of two 2D numpy arrays called respectively inputs and outputs.
    Their rows represent datapoints. inputs' columns represent the features of the datapoints. outputs' columns
    represent output variables. If the outputs are not know, set the variable to None.

    This class can also load the data on the GPU by transforming it into theano shared variables.
    You can load/save the dataset from/in hdf5 files.

    You can inherit from this class if you want to specify your own data attributes. If so, you may want to overwrite
    these methods: shuffle_data_virtual, write_virtual, read_virtual, copy_dataset_slice_virtual

    Attributes:
        inputs (2D numpy array): rows represent datapoints and columns represent features
        outputs (2D numpy array): if known, corresponding outputs of the inputs. rows represent datapoints and columns
            represent output vrariables.
        n_in_features (int): number of input featurse
        n_out_features (int): number of output features
        n_data (int): number of datapoints
        is_perm (boolean): indicates if the dataset is shuffled or not

    """
    def __init__(self):

        self._inputs = None
        self._inputs_shared = None
        self.n_in_features = None

        self._outputs = None
        self._outputs_shared = None
        self.n_out_features = None

        self.n_data = None
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

    def shuffle_data(self):
        """
        Shuffle the dataset.
        """
        perm = np.random.permutation(self.n_data)
        self.inputs = self.inputs[perm]
        self.outputs = self.outputs[perm]
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

        h5file.create_dataset("outputs", data=self.outputs, dtype='f')
        h5file.attrs['n_out_features'] = self.n_out_features

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
        self.outputs = h5file["outputs"].value
        self.read_virtual(h5file)
        h5file.close()

    def read_virtual(self, h5file):
        pass

    def split_dataset(self, proportion):
        """
        Split a dataset in two sub datasets according to proportion.
        """
        if proportion < 0 or proportion > 1:
            raise Exception("the proportion to split the dataset should be between 0 and 1")

        split = int(proportion * self.n_data)
        ds1 = self.duplicate_dataset_slice(slice(0, split))
        ds2 = self.duplicate_dataset_slice(slice(split, None))
        return ds1, ds2

    def duplicate_dataset_slice(self, slice_idx):
        """
        Create a Dataset that correspond to a slice of the Dataset calling the function.
        """
        ds = type(self)()
        ds.inputs = self.inputs[slice_idx]
        ds.outputs = self.outputs[slice_idx]
        self.duplicate_dataset_slice_virtual(ds, slice_idx)
        return ds

    def duplicate_dataset_slice_virtual(self, ds, slice_idx):
        """
        In case there are more attributes to slice.
        ds.your_attribute = self.your_attribute[slice_idx]
        """
        pass