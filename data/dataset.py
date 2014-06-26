__author__ = 'adeb'


from datetime import datetime

import h5py
import numpy as np

from spynet.utils.utilities import open_h5file


class Dataset():
    """
    Abstract class to create, store, save, load a dataset.

    Attributes:
        inputs (2D array): rows represent datapoints and columns represent features
        outputs (2D array): if known, corresponding outputs of the inputs
        n_in_features (int): number of input featurse
        n_out_features (int): number of output features
        n_data (int): number of datapoints
        is_perm (boolean): indicates if the dataset is shuffled or not
    """
    def __init__(self):
        self.inputs = None
        self.outputs = None

        self.n_in_features = None
        self.n_out_features = None
        self.n_data = None

        self.is_perm = False

    def populate_common(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.n_data, self.n_in_features = inputs.shape
        self.n_out_features = outputs.shape[1]

    def shuffle_data(self):
        """
        Shuffle the dataset.
        """
        perm = np.random.permutation(self.n_data)
        self.inputs = self.inputs[perm]
        self.outputs = self.outputs[perm]
        self.shuffle_data_virtual(perm)

    def shuffle_data_virtual(self, perm):
        pass

    def write(self, file_path):
        """
        write the dataset in a hdf5 file.
        """
        h5file = h5py.File(file_path, "w")
        h5file.create_dataset("inputs", data=self.inputs, dtype='f')
        h5file.create_dataset("outputs", data=self.outputs, dtype='f')

        h5file.attrs['creation_date'] = str(datetime.now())
        h5file.attrs['n_data'] = self.n_data
        h5file.attrs['n_in_features'] = self.n_in_features
        h5file.attrs['n_out_features'] = self.n_out_features
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
        self.outputs = h5file["outputs"].value

        self.n_data = int(h5file.attrs["n_data"])
        self.n_in_features = int(h5file.attrs["n_in_features"])
        self.n_out_features = int(h5file.attrs["n_out_features"])
        self.is_perm = bool(h5file.attrs['is_perm'])

        self.read_virtual(h5file)

        h5file.close()

    def read_virtual(self, h5file):
        pass