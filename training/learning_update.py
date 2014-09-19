__author__ = 'adeb'

from collections import OrderedDict
import numpy as np

import theano

from spynet.utils.utilities import share


class LearningUpdate():
    """
    Abstract class defining the update in a Trainer object.
    """
    def __init__(self):
        pass

    def compute_updates(self, params, grads):
        raise NotImplementedError

    @staticmethod
    def factory(**kwargs):
        """
        Factory function to create a learning update object from a dictionary.
        """
        update_type = kwargs["type"]
        learning_rate = kwargs["learning_rate"]
        if update_type == "GD":
            learning_update = LearningUpdateGD(learning_rate)
        elif update_type == "GDmomentum":
            learning_update = LearningUpdateGDMomentum(learning_rate, kwargs["momentum"])
        else:
            raise Exception("No Learning update with this name. Check the config file.")

        return learning_update


class LearningUpdateGD(LearningUpdate):
    """
    Gradient descent (GD) update.
    """
    def __init__(self, learning_rate):
        LearningUpdate.__init__(self)
        self.learning_rate = share(learning_rate, "learning_rate")

    def compute_updates(self, params, grads):
        updates = []
        for param_i, grad_i in zip(params, grads):
            updates.append((param_i, param_i - self.learning_rate * grad_i))
        return updates


class LearningUpdateGDMomentum(LearningUpdate):
    """
    GD + momentum.
    """
    def __init__(self, learning_rate, momentum):
        LearningUpdate.__init__(self)
        self.learning_rate = share(learning_rate, "learning_rate")
        self.momentum = share(momentum, "momentum")
        if momentum < 0 or momentum > 1:
            raise Exception("Momentum value should be between 0 and 1.")

    def compute_updates(self, params, grads):
        updates = OrderedDict()
        for param_i, grad_i in zip(params, grads):
            diff = share(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX), "diff")
            update_diff = self.momentum * diff - self.learning_rate * grad_i
            updates[param_i] = param_i + update_diff
            updates[diff] = update_diff
        return updates