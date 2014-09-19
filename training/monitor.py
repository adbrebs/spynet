__author__ = 'adeb'

from itertools import cycle
import numpy as np
from bisect import bisect
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save
import theano
import theano.tensor as T
from spynet.utils.utilities import error_rate, compute_dice, count_common_classes, compute_dice_from_counts


class Monitor():
    """
    Abstract class to monitor a specific statistic related to the training of a network.

    Attributes:
        type (string): type name of the monitor (defined by the class)
        name (string): name of the monitor (defined by the user)

        ds (Dataset object): dataset that the Monitor monitors

        n_batches_per_interval (int): number of batches between each measurement of the monitor

        ls_stopping_criteria (list of StoppingCriterion objects): a Monitor object may potentially be connected to
            StoppingCriterion objects. When a new measurement is recorded, the monitor sends a message to the
            corresponding stopping criteria.

        param_selector (ParamSelector object): a Monitor might be connected to a ParamSelector object. When a new
            measurement is recorded, the monitor sends a message to the corresponding ParamSelector.

        history_index (list of integers): list of the id_minibatch at which the monitor recorded a value
        history_value (list of reals): list of the values monitored at the corresponding id_minibatch of history_index
    """

    type = None

    def __init__(self, n_times_per_epoch, name, ds):
        self.name = name

        self.ds = ds

        self.n_times_per_epoch = n_times_per_epoch

        self.n_batches_per_interval = None

        self.ls_stopping_criteria = []

        self.history_minibatch = []
        self.history_value = []

        self.param_selector = None

        self.batch_size = 1000  # Forced to have a small batch size to prevent memory problems
        self.n_batches, self.last_batch_size = divmod(ds.n_data, self.batch_size)

        self.compute_batch_pred = None
        self.compute_last_batch_pred = None

    def init(self, trainer):

        self.n_batches_per_interval = int(trainer.n_train_batches / self.n_times_per_epoch)

        in_batch = T.matrix('in_batch')  # Minibatch input matrix
        tg_batch = T.matrix('tg_batch')  # True output (target) of a minibatch
        # Predicted output of the network for an input batch
        pred_batch = trainer.net.forward(in_batch, self.batch_size)

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        self.compute_batch_pred = theano.function(
            inputs=[idx_batch],
            outputs=T.argmax(pred_batch, axis=1),
            givens={in_batch: self.ds.inputs_shared[id1:id2]})
        if self.last_batch_size > 0:
            pred_batch = trainer.net.forward(in_batch, self.last_batch_size)
            self.compute_last_batch_pred = theano.function(
                inputs=[],
                outputs=T.argmax(pred_batch, axis=1),
                givens={in_batch: self.ds.inputs_shared[self.ds.n_data-self.last_batch_size:]})

    def add_stopping_criteria(self, ls_stopping_criteria):
        self.ls_stopping_criteria.extend(ls_stopping_criteria)

    def set_param_selector(self, param_selector):
        self.param_selector = param_selector

    def record(self, epoch, epoch_minibatch, id_minibatch, force_record=False, update_stopping=True, verbose=True):
        """
        Record a value of the monitored statistic of the training.
        Args:
            epoch, minibatch_idx, id_minibatch (int): state of the training
        """
        if (not force_record) and epoch_minibatch % self.n_batches_per_interval != 0:
            return False

        # Compute the monitored statistic
        measurement = self.compute_value()

        # Save value in history
        self.history_minibatch.append(id_minibatch)
        self.history_value.append(measurement)

        # Possibly select current parameters of the network
        if self.param_selector is not None:
            self.param_selector.update(id_minibatch, measurement)

        # Update possible stopping criteria connected to this monitor
        if update_stopping:
            for s_c in self.ls_stopping_criteria:
                s_c.update(epoch, epoch_minibatch, id_minibatch, verbose)

        return True

    def compute_value(self):
        """
        Compute the monitored statistic.
        """
        raise NotImplementedError

    def str_value_from_position(self, history_position):
        """
        Return the value of a specific record given its index in the history.
        """
        return "[{} {}: {}]".format(self.type, self.name, self.history_value[history_position])

    def str_value_from_minibatch(self, minibatch_idx):
        """
        Return the value of a specific record given its index in the history.
        """
        idx = self.history_minibatch.index(minibatch_idx)
        return self.str_value_from_position(idx)

    def get_minimum(self):
        minimum = np.inf
        for it, val in zip(self.history_minibatch, self.history_value):
            if val < minimum:
                minimum = val

        return minimum

    def get_maximum(self):
        maximum = -np.inf
        for it, val in zip(self.history_minibatch, self.history_value):
            if val > maximum:
                maximum = val

        return maximum

    @staticmethod
    def is_a_better_than_b(a, b, rate=1):
        return None


class MonitorErrorRate(Monitor):
    """
    Monitor that tracks the error rate of the network on a particular dataset
    """
    type = "Error rate"

    def __init__(self, n_times_per_epoch, name, ds):
        Monitor.__init__(self, n_times_per_epoch, name, ds)

    def compute_value(self):
        value = 0
        for i in xrange(self.n_batches):
            pred = self.compute_batch_pred(i)
            id1 = i * self.batch_size
            id2 = (i + 1) * self.batch_size
            value += error_rate(pred, np.argmax(self.ds.outputs[id1:id2], axis=1)) * self.batch_size
        if self.last_batch_size > 0:
            pred = self.compute_last_batch_pred()
            tg = np.argmax(self.ds.outputs[self.ds.n_data-self.last_batch_size:], axis=1)
            value += error_rate(pred, tg) * self.last_batch_size
        return value / self.ds.n_data

    @staticmethod
    def is_a_better_than_b(a, b, rate=1):
        return a < (b*rate)


class MonitorDiceCoefficient(Monitor):
    """
    Monitor that tracks the dice coefficient of the network on a particular dataset
    """
    type = "Dice coefficient"

    def __init__(self, n_times_per_epoch, name, ds, n_classes):
        self.n_classes = n_classes
        Monitor.__init__(self, n_times_per_epoch, name, ds)

    def compute_value(self):
        counts = np.zeros((self.n_classes-1, 3))
        for i in xrange(self.n_batches):
            pred = self.compute_batch_pred(i)
            id1 = i * self.batch_size
            id2 = (i + 1) * self.batch_size
            counts += count_common_classes(pred, np.argmax(self.ds.outputs[id1:id2], axis=1), self.n_classes)
        if self.last_batch_size > 0:
            pred = self.compute_last_batch_pred()
            tg = np.argmax(self.ds.outputs[self.ds.n_data-self.last_batch_size:], axis=1)
            counts += count_common_classes(pred, tg, self.n_classes)

        return np.mean(compute_dice_from_counts(counts))

    @staticmethod
    def is_a_better_than_b(a, b, rate=1):
        return (a*rate) > b


def save_records_plot(file_path, ls_monitors, name, n_train_batches, legend_loc="upper right"):
    """
    Save a plot of a list of monitors' history.
    Args:
        file_path (string): the folder path where to save the plot
        ls_monitors: the list of statistics to plot
        name: name of file to be saved
        n_train_batches: the total number of training batches
    """

    lines = ["--", "-", "-.",":"]
    linecycler = cycle(lines)

    plt.figure()
    for m in ls_monitors:
        X = [i/float(n_train_batches) for i in m.history_minibatch]
        Y = m.history_value
        a, b = zip(*sorted(zip(X, Y)))
        plt.plot(a, b, next(linecycler), label=m.name)

    plt.xlabel('Training epoch')
    plt.ylabel(ls_monitors[0].type)
    plt.legend(loc=legend_loc)
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=10)
    plt.savefig(file_path + name + ".png")
    tikz_save(file_path + name + ".tikz", figureheight = '\\figureheighttik', figurewidth = '\\figurewidthtik')