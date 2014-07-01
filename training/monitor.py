__author__ = 'adeb'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
from spynet.utils.utilities import error_rate_symb


class Monitor():
    """
    Abstract class to monitor a specific statistic related to the training.
    Attributes:
        type (string): type of the monitor (defined by the class)
        name (string): name of the monitor (defined by the user)

        trainer (Trainer object): Trainer object monitored by the monitor

        n_batches_per_interval (int): number of batches between monitoring

        ls_stopping_criteria (list of StoppingCriterion objects): a Monitor object may potentially be connected to
            StoppingCriterion objects.

        history (list of 2-tuples): list of tuples (id_minibatch, monitored_value for this minibatch) representing the
            history of records of the monitor
    """
    def __init__(self, trainer, n_times_per_epoch, name):
        self.type = None
        self.name = name

        self.trainer = trainer

        self.n_batches_per_interval = int(trainer.n_train_batches / n_times_per_epoch)

        self.ls_stopping_criteria = []

        self.history = []

    def add_stopping_criteria(self, ls_stopping_criteria):
        self.ls_stopping_criteria.extend(ls_stopping_criteria)

    def record(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Record a value of the monitored statistic of the training.
        Args:
            epoch, minibatch_idx, id_minibatch (int): state of the training
        """
        if minibatch_idx % self.n_batches_per_interval != 0:
            return False

        # Compute the monitored statistic
        measurement = self.compute_value(epoch, minibatch_idx, id_minibatch, verbose)

        # Save value in history
        self.history.append((id_minibatch, measurement))

        # Update possible stopping criteria connected to this monitor
        for s_c in self.ls_stopping_criteria:
            s_c.update(epoch, minibatch_idx, id_minibatch, verbose)

        return True

    def compute_value(self, epoch, minibatch_idx, id_minibatch, verbose):
        """
        Compute the monitored statistic.
        """
        raise NotImplementedError

    def str_value(self, history_idx):
        """
        Return the value of a specific record given its index in the history.
        """
        return "[{} {}: {}]".format(self.type, self.name, self.history[history_idx][1])


class MonitorErrorRate(Monitor):
    """
    Monitor that tracks the error rate of the network on a particular dataset
    """
    def __init__(self, trainer, n_times_per_epoch, name, ds):
        Monitor.__init__(self, trainer, n_times_per_epoch, name)
        self.type = "Error rate"

        self.ds = ds

        in_batch = T.matrix('in_batch')  # Minibatch input matrix
        tg_batch = T.matrix('tg_batch')  # True output (target) of a minibatch
        # Predicted output of the network for an input batch
        pred_batch = trainer.net.forward(in_batch, ds.n_data)

        self.errorRate = theano.function(
            inputs=[],
            outputs=error_rate_symb(pred_batch, tg_batch),
            givens={in_batch: ds.inputs_shared, tg_batch: ds.outputs_shared})

    def compute_value(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        return self.errorRate()


def save_records_plot(file_path, ls_monitors):
    """
    Save a plot of a list of monitors' history.
    Args:
        experiment_path (string): the folder path where to save the plot
    """

    def save_error(error, legend):
        plt.plot(*zip(*error), label=legend)

    for m in ls_monitors:
        save_error(m.history, m.name)

    plt.xlabel('Minibatch index')
    plt.ylabel(ls_monitors[0].type)
    plt.legend(loc='upper right')
    plt.savefig(file_path + "err.png")