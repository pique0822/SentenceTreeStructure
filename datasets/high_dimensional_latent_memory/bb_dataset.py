import numpy as np

class Dataset(object):
    """Defines a dataset for the sequentially parsing through bounding box data."""

    def __init__(self, dim, num_training, num_testing, repeated_sequences=True, seed=0):
        """dim  - the number of dimensions we want the model to learn and predict."""
        super(Dataset, self).__init__()
        self.dim = dim
        self.num_training = num_training
        self.num_testing = num_testing

    def generate_sequence(length=None):
        sequence_length = np.random.randint(100,500)

        np.random.rand(length,)
