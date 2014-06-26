__author__ = 'adeb'


from spynet.utils.utilities import share


class DataBase():
    """
    Abstract class responsible for splitting datasets into training, validating and testing datasets. It also loads
    them on the GPU. The functions that need to be defined by a child class are __init__ and create_datasets.

    Attributes:
        test_in, test_out, valid_in, valid_out, train_in, train_out (Theano shared 2D matrices): data sets loaded on
            the GPU.
    """
    def __init__(self):

        self.n_in_features = None
        self.n_out_features = None
        self.n_data = None

        self.test_in = None
        self.test_out = None
        self.valid_in = None
        self.valid_out = None
        self.train_in = None
        self.train_out = None

        self.n_train = None
        self.n_valid = None
        self.n_test = None
        self.prop_validation = None

    def load_datasets(self, config):
        # Has to be defined in children classes
        raise NotImplementedError

    def init(self, prop_validation, config=None):
        self.prop_validation = prop_validation

        # Create datasets
        training_data, testing_data = self.load_datasets(config)

        self.n_out_features = training_data.n_out_features
        n_data = training_data.n_data

        # Create a validation set
        prop_validation = prop_validation
        validatioin_split = int((1-prop_validation) * n_data)
        train_x = training_data.inputs[0:validatioin_split, :]
        train_y = training_data.outputs[0:validatioin_split, :]
        valid_x = training_data.inputs[validatioin_split:n_data, :]
        valid_y = training_data.outputs[validatioin_split:n_data, :]

        # Testing data
        test_x = testing_data.inputs
        test_y = testing_data.outputs

        # Transform the data into Theano shared variables
        self.share_data(test_x, test_y, valid_x, valid_y, train_x, train_y)

    def share_data(self, test_in, test_out, valid_in, valid_out, train_in, train_out):
        """
        Store the data in shared variables
        """
        self.test_in = share(test_in)
        self.test_out = share(test_out)
        self.valid_in = share(valid_in)
        self.valid_out = share(valid_out)
        self.train_in = share(train_in)
        self.train_out = share(train_out)

        self.n_train, self.n_in_features = train_in.shape
        self.n_valid = valid_in.shape[0]
        self.n_test = test_in.shape[0]

        self.n_out_features = train_out.shape[1]
        self.n_data = self.n_train + self.n_valid + self.n_test
