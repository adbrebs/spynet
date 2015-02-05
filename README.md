This library is not anymore maintained, you should rather have a look to the following ones:
- https://github.com/lisa-lab/pylearn2
- https://github.com/benanne/Lasagne
- https://github.com/bartvm/blocks

Neural Networks library on top of Theano
==================
            
Flexible research library designed to build feedforward neural networks with complicated graphs. In particular, it is easy to:
- use different neuron models and connections in a same layer
- have blocks of neurons with multiple inputs/outputs, merge/split them to create complicated feedforward architectures.
- share weights between different blocks of neurons
- monitor custom statistics during the training
- generate data on the fly while the training is running (not finished yet)

Requirements
--------------
Python packages that you need:
- theano
- h5py

Set up your own project
--------------
Create a folder for your project, say ```./yours/```, and copy the spynet folder in this ```./yours/``` folder 
so that the hierarchy looks like:
```python
./yours/spynet/*
```
Then you need to inherit from at least the two following classes:  
- Network (spynet.models.network.Network) where you define the architecture of your model  
- Experiment (spynet.experiment.Experiment) where you define your training experiment, i.e. all the instructions you
want to run. Its function Experiment.run() will be your main program.

You may potentially inherit from Dataset if you need to save additional information about your data:  
- Dataset (spynet.data.dataset.Dataset) which stores a numpy datasets of inputs and corresponding outputs  

Your project directory should now looks like that:  
```
./yours/spynet/*  
./yours/network_yours.py  # where your inherit from Network  
./yours/experiment_yours.py  # where you inherit from Experiment
```  

To write the code of your classes Network and Experiment, you take inspiration from MNIST example provided in the spynet
folder.

Get started with the MNIST example
--------------
An experiment example is provided in ```./yours/spynet/mnist_example/```.  
The experiment creates a convolutional neural network (similar to LeNet) and train it on the MNIST dataset 
(http://yann.lecun.com/exdb/mnist/).  
Just run the file ```./yours/spynet/mnist_example/experiment_mnist.py```.


Class Diagram
--------------
A simplified class diagram is sketched on this page: 
https://drive.google.com/file/d/0B7nfeKBWzl-heDJtTnJLNnhHWDA/edit?usp=sharing


Note: the mnist example and some chunks of code take inspiration from the theano tutorials on deeplearning.net
