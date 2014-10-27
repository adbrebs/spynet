__author__ = 'adeb'

# Hack to be able to run this module
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from shutil import copy2
import inspect
from spynet.data.dataset import Dataset
from spynet.models.network import *
from spynet.models.neuron_type import *
from spynet.training.trainer import *
from spynet.training.monitor import *
from spynet.training.parameters_selector import *
from spynet.training.stopping_criterion import *
from spynet.training.cost_function import *
from spynet.training.learning_update import *
from spynet.experiment import Experiment
from transform_mnist_to_h5 import transform_mnist_to_h5


class ExperimentMNIST(Experiment):
    def __init__(self, exp_name, data_path):
        Experiment.__init__(self, exp_name, data_path)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        training_data_path = self.data_path + "train.h5"
        testing_data_path = self.data_path + "test.h5"

        # If files don't already exist, create them
        if not os.path.isfile(training_data_path):
            transform_mnist_to_h5()

        prop_validation = 0.15  # Percentage of the training dataset that is used for validation (early stopping)
        ds_training = Dataset.create_and_read(training_data_path)
        ds_validation, ds_training = ds_training.split_dataset_proportions([prop_validation, 1-prop_validation])
        ds_testing = Dataset.create_and_read(testing_data_path)
        ds_training.outputs = ds_training.inputs
        ds_validation.outputs = ds_validation.inputs
        ds_testing.outputs = ds_testing.inputs

        # Scale the data
        # s = Scaler([slice(None, None)])
        # s.compute_parameters(ds_training.inputs)
        # s.scale(ds_training.inputs)
        # s.scale(ds_validation.inputs)
        # s.scale(ds_testing.inputs)

        ###### Create the network

        # net = NetworkMNIST()
        net = AutoEncoder()
        net.init([28**2, 256, 28**2], dropout=True, dropout_p=[0.5], neuron_function=NeuronRELU())
        print net

        ###### Configure the trainer

        # Cost function
        cost_function = CostMSE()

        # Learning update
        learning_rate = 0.01
        momentum = 0.5
        lr_update = LearningUpdateGDMomentum(learning_rate, momentum)

        # Create monitors and add them to the trainer
        err_training = MonitorMSE(1, "Training", ds_training)
        err_testing = MonitorMSE(1, "Testing", ds_testing)
        err_validation = MonitorMSE(1, "Validation", ds_validation)

        # Create stopping criteria and add them to the trainer
        max_epoch = MaxEpoch(15)
        early_stopping = EarlyStopping(err_validation)

        # Create the network selector
        params_selector = ParamSelectorBestMonitoredValue(err_validation)

        # Create the trainer object
        batch_size = 20
        t = Trainer(net, cost_function, params_selector, [max_epoch, early_stopping],
                    lr_update, ds_training, batch_size,
                    [err_training, err_testing, err_validation])

        ###### Train the network

        t.train()

        ###### Plot the records

        save_records_plot(self.path, [err_training, err_testing, err_validation], "errors", t.n_train_batches)

        ###### Save the network

        net.save_parameters(self.path + "netdrop.net")


if __name__ == '__main__':

    exp_name = "mnist_example"
    data_path = "./datasets/mnist/"

    exp = ExperimentMNIST(exp_name, data_path)
    exp.run()