__author__ = 'adeb'

import numpy as np


class ParamSelector():
    """
    These classes determine on which basis the final trained network is selected.
    """
    def __init__(self):
        self.best_params = None
        self.best_iter = None
        self.net = None

    def init(self, net):
        self.net = net

    def update(self, iteration, monitored_value):
        raise NotImplementedError

    def update_network(self):
        self.net.import_params(self.best_params)


class ParamSelectorBestMonitoredValue(ParamSelector):
    """
    The final network is the one with the best monitored value.
    """
    def __init__(self, monitor):
        ParamSelector.__init__(self)
        monitor.set_param_selector(self)
        self.monitor = monitor

        if self.monitor.is_a_better_than_b(2,1):
            self.best_monitored_value = -np.inf
        else:
            self.best_monitored_value = np.inf

    def update(self, iteration, monitored_value):
        if self.monitor.is_a_better_than_b(monitored_value, self.best_monitored_value):
            self.best_monitored_value = monitored_value
            self.best_params = self.net.export_params()
            self.best_iter = iteration