__author__ = 'adeb'

import cPickle
import gzip
import os

import numpy

import h5py

dataset_gz = "./mnist.pkl.gz"
output_path = "./datasets/mnist/"

# If data not here
if not os.path.isfile(dataset_gz):
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, dataset_gz)

# Check if output path exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

##################################

data_dir, data_file = os.path.split(dataset_gz)
if data_dir == "" and not os.path.isfile(dataset_gz):
    # Check if dataset is in the data directory.
    new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset_gz)
    if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
        dataset_gz = new_path

print "Transforming mnist data ..."

# Load the dataset
f = gzip.open(dataset_gz, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

n_classes = 10

train_x = numpy.concatenate((train_set[0], valid_set[0]))
train_y0 = numpy.concatenate((train_set[1], valid_set[1]))
train_y = numpy.zeros((train_x.shape[0], n_classes))
train_y[range(train_x.shape[0]), train_y0] = 1

test_x = test_set[0]
test_y0 = test_set[1]
test_y = numpy.zeros((test_x.shape[0], n_classes))
test_y[range(test_x.shape[0]), test_y0] = 1

f = h5py.File(output_path + 'train.h5', 'w')
f['/inputs'] = train_x
f['/outputs'] = train_y
f.attrs["n_data"] = train_x.shape[0]
f.attrs['n_out_features'] = n_classes
f.attrs["n_in_features"] = train_x.shape[1]
f.attrs['is_perm'] = True
f.close()


f = h5py.File(output_path + 'test.h5', 'w')
f['/inputs'] = test_x
f['/outputs'] = test_y
f.attrs["n_data"] = test_x.shape[0]
f.attrs['n_out_features'] = n_classes
f.attrs["n_in_features"] = test_x.shape[1]
f.attrs['is_perm'] = True
f.close()

