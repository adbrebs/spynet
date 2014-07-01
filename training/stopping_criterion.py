__author__ = 'adeb'

import numpy as np


class StoppingCriterion():
    """
    Abstract class defining a stopping criterion for the trainer
    """
    def __init__(self, trainer):
        self.trainer = trainer

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Check if the stopping criterion is triggered or not.
        """
        raise NotImplementedError


class MaxEpoch(StoppingCriterion):
    """
    Stopping criterion that triggers when a maximal number of epochs is reached
    """
    def __init__(self, trainer, max_epoch):
        StoppingCriterion.__init__(self, trainer)
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
    def __init__(self, trainer, monitor, patience_increase=5, improvement_threshold=0.99, initial_patience=5):
        StoppingCriterion.__init__(self, trainer)
        self.monitor = monitor

        self.patience_increase = patience_increase
        self.improvement_threshold = improvement_threshold
        self.patience = initial_patience

        # Link the monitor to the stopping criterion
        self.monitor.add_stopping_criteria([self])

        # Save the best params
        self.best_params = None
        self.best_monitor_value = np.inf
        self.best_iter = 0

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
        (id_monitoring, monitored_value) = self.monitor.history[-1]

        if monitored_value > self.best_monitor_value * self.improvement_threshold:
            return

        # Increase the patience is the value has sufficiently increased
        self.patience = epoch + self.patience_increase

        # save the monitored value and the corresponding parameters of the network
        self.best_monitor_value = monitored_value
        self.best_iter = id_monitoring
        self.best_params = self.trainer.net.export_params()

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        # In case we stop, loads the best parameters
        if self.stopping:
            if verbose:
                print("Stopping criterion triggered: out of patience")
            self.trainer.net.import_params(self.best_params)

        return self.stopping