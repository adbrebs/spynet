__author__ = 'adeb'

import time

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
    def __init__(self, net, cost_function, learning_update, ds_training, batch_size):
        print 'Configure training ...'

        self.net = net
        self.ds_training = ds_training

        self.ls_monitors = []
        self.ls_stopping_criteria = []

        self.batch_size = batch_size
        self.n_train_batches = ds_training.n_data / self.batch_size

        self.in_batch = in_batch = T.matrix('in_batch')  # Minibatch input matrix
        self.tg_batch = tg_batch = T.matrix('tg_batch')  # True output (target) of a minibatch
        # Predicted output of the network for an input batch
        self.pred_batch = pred_batch = net.forward(in_batch, self.batch_size)

        # Cost the trainer is going to minimize
        self.cost_function = cost_function
        cost = cost_function.compute_cost_symb(pred_batch, tg_batch)

        # Compute gradients
        params = net.ls_params
        grads = T.grad(cost, params)

        # Compute updates
        self.learning_update = learning_update
        updates = learning_update.compute_updates(params, grads)

        idx_batch = T.lscalar()
        id1 = idx_batch * self.batch_size
        id2 = (idx_batch + 1) * self.batch_size
        in_train = ds_training.inputs_shared
        out_train = ds_training.outputs_shared
        self.train_minibatch = theano.function(
            inputs=[idx_batch],
            outputs=cost,
            updates=updates,
            givens={in_batch: in_train[id1:id2], tg_batch: out_train[id1:id2]})

    def add_monitors(self, ls_monitors):
        self.ls_monitors.extend(ls_monitors)

    def add_stopping_criteria(self, ls_stopping_criteria):
        self.ls_stopping_criteria.extend(ls_stopping_criteria)

    def check_if_stop(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Check if the training should stop
        """
        for stopping_cri in self.ls_stopping_criteria:
            if stopping_cri.check_if_stop(epoch, minibatch_idx, id_minibatch, verbose):
                return True
        return False

    def record(self, epoch, minibatch_idx, id_minibatch, verbose=True):
        """
        Record statistics about the training.
        Returns True is at least one value is recorded.
        """
        updated_monitors = []  # memorize monitors that record a new value
        for i, monitor in enumerate(self.ls_monitors):
            if monitor.record(epoch, minibatch_idx, id_minibatch):
                updated_monitors.append(i)

        if verbose and updated_monitors:
            print("    minibatch {}/{}:".format(minibatch_idx, self.n_train_batches))
            for i in updated_monitors:
                print("        {}".format(self.ls_monitors[i].str_value(-1)))

        if updated_monitors:
            return True
        else:
            return False

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
            for minibatch_index in xrange(1, 1+self.n_train_batches):

                minibatch_id += 1

                # Display minibatch number
                if minibatch_id % freq_display_batch == 0:
                    print("    minibatch {}/{}".format(minibatch_index, self.n_train_batches))

                # Train on the current minibatch
                self.train_minibatch(minibatch_index-1)

                # Record statistics
                self.record(epoch_id, minibatch_index, minibatch_id)

                # Check if a stopping criterion is met
                if self.check_if_stop(epoch_id, 0, minibatch_id):
                    stop = True
                    break

            if not stop:
                print("    epoch {} finished after {} seconds".format(epoch_id, time.clock() - starting_epoch_time))

        end_time = time.clock()
        print ("Training ran for {} minutes".format((end_time - start_time) / 60.))