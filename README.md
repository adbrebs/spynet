Neural Networks library on top of Theano
==================

Flexible research library implementing basic building blocks of neural networks. In particular, it is easy to:
- use different neuron models and connections in a same layer
- share weights between different parts of the architecture
- generate data on the fly while the training is running (not finished yet)

Requirements
--------------
Python packages that you need:
- theano
- h5py

Setup
--------------

Create a folder for your project, say /mnist/, and copy the nn folder in this /mnist/ folder so that the hierarchy looks like:
```python
./mnist/nn/*
```
Then you need to inherit from at least two classes of the library:
- Network (nn.models.network.Network) where you define the architecture of your model
- DataBase (nn.data.database.DataBase) where you define your training, testing and validation datasets

You also need to create your own configuration file for the training. The config file model is found in ./nn/cfg_training_model.py, copy it into ./mnist/ and rename it to cfg_training.py so your project directory now looks like that:  
```python
./mnist/nn/*  
./mnist/network_mnist.py  # where your inherit from Network  
./mnist/database_mnist.py  # where you inherit from DataBase  
./mnist/cfg_training.py  # a copy of ./nn/cfg_training_model.py with your own configuration
```
In order to create your DataBase class, you might find convenient to also inherit from Dataset and DataGenerator.

Once your classes are defined, you can create a main_training.py file to train your model. For instance, this file can look like:  
```python
from nn.utils.utilities import load_config
from nn.training.trainer import Trainer
from network_mnist import NetworkMNIST
from database_mnist import DataBaseMNIST

if __name__ == '__main__':

    ### Load the config file
    training_cf = load_config("cfg_general")

    ### Create the database
    db = DataBaseMNIST()
    db.init(training_cf.prop_validation, training_cf)

    ### Create the network
    net = NetworkMNIST()
    net.init(n_in=28**2, n_out=10)
    print net

    ### Train the network
    t = Trainer(training_cf, net, db)
    t.train()

    ### Save the network
    net.save_parameters(training_cf.net_path)
```
