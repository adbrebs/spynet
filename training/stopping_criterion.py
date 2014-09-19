__author__ = 'adeb'

import numpy as np


class StoppingCriterion():
    """
    Abstract class defining a stopping criterion for the trainer
    """
    def __init__(self):
        pass

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Check if the stopping criterion is triggered or not.
        """
        raise NotImplementedError


class MaxEpoch(StoppingCriterion):
    """
    Stopping criterion that triggers when a maximal number of epochs is reached
    """
    def __init__(self, max_epoch):
        StoppingCriterion.__init__(self)
        self.max_epoch = max_epoch

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        if epoch < self.max_epoch:
            return False

        if verbose:
            print("Stopping criterion triggered: maximum number of epoch reached")
        return True


class EarlyStopping(StoppingCriterion):
    """
    Stopping criterion monitoring a monitor. When the monitored value changes, the monitor updates the EarlyStopping
    object.
    """
    def __init__(self, monitor, patience_increase=5, improvement_threshold=0.99, initial_patience=5):
        StoppingCriterion.__init__(self)
        self.monitor = monitor

        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.patience = initial_patience

        # Link the monitor to the stopping criterion
        self.monitor.add_stopping_criteria([self])

        # Save the best monitored value
        if self.monitor.is_a_better_than_b(2,1):
            self.best_monitor_value = -np.inf
        else:
            self.best_monitor_value = np.inf

        # Indicates if the stopping criterion is triggered or not
        self.stopping = False

    def update(self, epoch, minibatch_idx, id_minibatch, verbose):
        """
        This function is called by a the Monitor object that the Stopping Criterion monitors.
        """

        ### Triggered
        if self.patience <= epoch:
            self.stopping = True
            return

        ### Not triggered yet
        # Fetch the monitor value
        (id_monitoring, monitored_value) = (self.monitor.history_minibatch[-1], self.monitor.history_value[-1])

        if ~self.monitor.is_a_better_than_b(monitored_value, self.best_monitor_value, self.improvement_threshold):
            return

        # Increase the patience is the value has sufficiently increased
        self.patience = epoch + self.patience_increase
        print("         patience increased")

        # save the monitored value and the corresponding parameters of the network
        self.best_monitor_value = monitored_value

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        # In case we stop, loads the best parameters
        if self.stopping and verbose:
            print("Stopping criterion triggered: out of patience")

        return self.stopping