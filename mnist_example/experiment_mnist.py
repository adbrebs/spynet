__author__ = 'adeb'

# Hack to be able to run this module
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

from shutil import copy2
import inspect
from spynet.utils.utilities import analyse_classes
from spynet.data.dataset import Dataset, Scaler
from spynet.mnist_example.network_mnist import NetworkMNIST
from spynet.training.trainer import *
from spynet.training.monitor import *
from spynet.training.parameters_selector import *
from spynet.training.stopping_criterion import *
from spynet.training.cost_function import *
from spynet.training.learning_update import *
from spynet.experiment import Experiment
from transform_mnist_to_h5 import transform_mnist_to_h5


class ExperimentMNIST(Experiment):
    def __init__(self, name):
        Experiment.__init__(self, name)

    def copy_file_virtual(self):
        copy2(inspect.getfile(inspect.currentframe()), self.path)

    def run(self):
        ###### Create the datasets

        data_path = "./datasets/mnist/"
        training_data_path = data_path + "train.h5"
        testing_data_path = data_path + "test.h5"
        if not os.path.isfile(training_data_path):
            transform_mnist_to_h5()
        prop_validation = 0.3  # Percentage of the training dataset that is used for validation (early stopping)
        ds_training = Dataset()
        ds_training.read(training_data_path)
        ds_validation, ds_training = ds_training.split_datapoints_proportion(prop_validation)
        ds_testing = Dataset()
        ds_testing.read(testing_data_path)
        # Few stats about the targets
        analyse_classes(np.argmax(ds_training.outputs, axis=1))

        # Scale the data
        s = Scaler([slice(None, None)])
        s.compute_parameters(ds_training.inputs)
        s.scale(ds_training.inputs)
        s.scale(ds_validation.inputs)
        s.scale(ds_testing.inputs)

        ###### Create the network

        net = NetworkMNIST()
        net.init(28, 28, 10)
        print net

        ###### Configure the trainer

        # Cost function
        cost_function = CostNegLL()

        # Learning update
        learning_rate = 0.13
        momentum = 0.5
        lr_update = LearningUpdateGDMomentum(learning_rate, momentum)

        # Create monitors and add them to the trainer
        err_training = MonitorErrorRate(1, "Training", ds_training)
        err_testing = MonitorErrorRate(1, "Testing", ds_testing)
        err_validation = MonitorErrorRate(1, "Validation", ds_validation)

        # Create stopping criteria and add them to the trainer
        max_epoch = MaxEpoch(300)
        early_stopping = EarlyStopping(err_validation)

        # Create the network selector
        params_selector = ParamSelectorBestMonitoredValue(err_validation)

        # Create the trainer object
        batch_size = 200
        t = Trainer(net, cost_function, params_selector, [max_epoch, early_stopping],
                    lr_update, ds_training, batch_size,
                    [err_training, err_testing, err_validation])

        ###### Train the network

        t.train()

        ###### Plot the records

        save_records_plot(self.path, [err_training, err_testing, err_validation], "errors", t.n_train_batches)

        ###### Save the network

        net.save_parameters(self.path + "net.net")


if __name__ == '__main__':

    exp = ExperimentMNIST("mnist_example")
    exp.run()