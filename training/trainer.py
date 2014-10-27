__author__ = 'adeb'

import time
import numpy as np

import theano
import theano.tensor as T


class Trainer():
    """
    Class that supervises the training of a neural network.

    Attributes:
        net (Network object): the network to be trained
        ds_training (Dataset object): the dataset on which the network is trained
        cost_function (CostFunction object): the cost function of the training

        batch_size (int): number of training datapoints to include in a training batch
        n_train_batches (int): number of batches that the dataset contains

        ls_monitors (list of Monitor objects): each monitor tracks a particular statistic of the training
        ls_stopping_criteria (list of StoppingCriterion objects): stopping criteria that decide when to
            stop the training

        train_minibatch (function): function to train the network on a single minibatch
    """
    def __init__(self, net, cost_function, params_selector, ls_stopping_criteria,
                 learning_update, ds_training, batch_size, ls_monitors):
        print 'Configure training ...'

        self.net = net
        self.cost_function = cost_function
        self.params_selector = params_selector
        self.ds_training = ds_training
        self.ls_monitors = ls_monitors
        self.ls_stopping_criteria = ls_stopping_criteria
        self.learning_update = learning_update
        self.batch_size = batch_size
        self.n_train_batches = ds_training.n_data / self.batch_size

        self.train_minibatch = None

        self.init()

    # Allow to reinitialize the object
    def init(self):
        for monitor in self.ls_monitors:
            monitor.init(self)

        self.params_selector.init(self.net)

        # Minibatch input matrix
        in_batch = T.matrix('in_batch')
        # True output (target) of a minibatch
        tg_batch = T.matrix('tg_batch')
        # Predicted output of the network for an input batch
        pred_batch = self.net.forward(in_batch, self.batch_size, False)

        # Cost the trainer is going to minimize
        cost = self.cost_function.compute_cost_symb(pred_batch, tg_batch)

        # Compute gradients
        params = self.net.ls_params
        grads = T.grad(cost, params)

        # Compute updates
        updates = self.learning_update.compute_updates(params, grads)

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        in_train = self.ds_training.inputs_shared
        out_train = self.ds_training.outputs_shared
        self.train_minibatch = theano.function(
            inputs=[idx_batch],
            outputs=cost,
            updates=updates,
            givens={in_batch: in_train[id1:id2], tg_batch: out_train[id1:id2]})

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Check if the training should stop
        """
        for stopping_cri in self.ls_stopping_criteria:
            if stopping_cri.check_if_stop(epoch, minibatch_idx, id_minibatch, verbose):
                return True
        return False

    def record(self, epoch, epoch_minibatch, id_minibatch, force_record=False, update_stopping=True, verbose=True):
        """
        Record statistics about the training.
        Returns True is at least one value is recorded.
        """
        updated_monitors = []  # memorize monitors that record a new value

        for i, monitor in enumerate(self.ls_monitors):
            has_monitored = monitor.record(epoch, epoch_minibatch, id_minibatch, force_record, update_stopping, verbose)
            if has_monitored:
                updated_monitors.append(i)

        if verbose and updated_monitors:
            print("    minibatch {}/{}:".format(epoch_minibatch, self.n_train_batches))
            for i in updated_monitors:
                print("        {}".format(self.ls_monitors[i].str_value_from_position(-1)))

    def train(self):
        print "Train the network ..."

        start_time = time.clock()

        freq_display_batch = max(self.n_train_batches / 4, 1)  # Frequency for printing the batch id
        epoch_id = minibatch_id = 0

        # Record statistics before training really starts
        self.record(epoch_id, 0, minibatch_id)

        stop = False
        while not stop:
            starting_epoch_time = time.clock()
            epoch_id += 1
            print("Epoch {}".format(epoch_id))

            for epoch_minibatch in xrange(1, 1+self.n_train_batches):

                minibatch_id += 1

                # Display minibatch number
                if epoch_minibatch % freq_display_batch == 0:
                    print("    minibatch {}/{}".format(epoch_minibatch, self.n_train_batches))

                # Train on the current minibatch
                if np.isnan(self.train_minibatch(epoch_minibatch-1)):
                    raise Exception("Error: The cost is Nan. Epoch {}, minibatch {}".format(epoch_id, epoch_minibatch))

                # Record statistics
                self.record(epoch_id, epoch_minibatch, minibatch_id)

                # Check if a stopping criterion is met
                if self.check_if_stop(epoch_id, 0, minibatch_id):
                    stop = True
                    break

            if not stop:
                print("    epoch {} finished after {} seconds".format(epoch_id, time.clock() - starting_epoch_time))

        end_time = time.clock()
        print ("Training ran for {} minutes".format((end_time - start_time) / 60.))
        # Update the network with the selected parameters
        self.params_selector.update_network()

        # Display monitored values for the best network
        self.record(None, None, self.params_selector.best_iter, force_record=True, update_stopping=False)