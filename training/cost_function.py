__author__ = 'adeb'

import math
import numpy as np
import theano.tensor as T


class CostFunction():
    """
    Cost function used during the training
    """
    def __init__(self):
        pass

    def compute_cost_symb(self, pred_batch, tg_batch):
        """
        Compute the cost symbolically with Theano tensors.
        Args:
            pred_batch (theano.tensor.TensorType): predicted output returned by the network
            tg_batch (theano.tensor.TensorType): output returned by the network
        Return:
            (theano.tensor.TensorType): a tensor representing the cost
        """
        raise NotImplementedError

    def compute_cost_numpy(self, pred_batch, tg_batch):
        """
        Compute the cost given numpy variables.
        """
        raise NotImplementedError

    @staticmethod
    def factory(**kwargs):
        """
        Factory function to create a cost function from a dictionary.
        """
        update_type = kwargs["type"]
        if update_type == "MSE":
            cost_function = CostMSE()
        elif update_type == "NLL":
            cost_function = CostNegLL()
        else:
            raise Exception("No cost function with this name. Check the config file.")

        return cost_function


class CostMSE(CostFunction):
    """
    Mean square error
    """
    def __init__(self):
        CostFunction.__init__(self)

    def compute_cost_symb(self, pred_batch, tg_batch):
        return T.mean(T.sum((pred_batch - tg_batch) * (pred_batch - tg_batch), axis=1))

    def compute_cost_numpy(self, pred_batch, tg_batch):
        return np.mean(np.sum((pred_batch - tg_batch) * (pred_batch - tg_batch), axis=1))


class CostNegLL(CostFunction):
    """
    Negative log-likelihood
    """
    def __init__(self):
        CostFunction.__init__(self)

    def compute_cost_symb(self, pred_batch, tg_batch):
        return -T.mean(T.log(T.sum(pred_batch * tg_batch, axis=1)))

    def compute_cost_numpy(self, pred_batch, tg_batch):
        return -np.mean(np.sum(math.log(pred_batch) * tg_batch, axis=1))