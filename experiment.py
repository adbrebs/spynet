__author__ = 'adeb'

from spynet.utils.utilities import create_directories


class Experiment():
    def __init__(self, exp_name, data_path):
        self.data_path = data_path
        self.name = exp_name
        self.path = create_directories(exp_name)
        self.copy_file_virtual()

    def copy_file_virtual(self):
        """
        Copy the experiment file into the experiment path
        This function has to be overwritten with the following code:
        copy2(inspect.getfile(inspect.currentframe()), self.path)
        """
        raise NotImplementedError

    def run(self):
        raise NotImplementedError