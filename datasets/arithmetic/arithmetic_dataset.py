import numpy as np

class Dataset(object):
    """Defines a dataset for the artihmetic data."""

    def __init__(self, training_file, testing_file):
        super(Dataset, self).__init__()
        self.training_file = training_file
        self.testing_file = testing_file

        self.training_x,self.training_y = self.file_to_list_of_strings(self.training_file)
        self.testing_x,self.testing_y = self.file_to_list_of_strings(self.testing_file)

        self.training_order = self.__shuffle__(range(len(self.training_x)))

        self.testing_order = self.__shuffle__(range(len(self.testing_x)))

        self.vector_size = 13

    def character_to_vector(self, character):
        """Possible characters are the number 0-9,-,+,= therefore 13 characters."""

        one_hot_encoding = np.zeros(self.vector_size)

        if character == '-':
            index = 10
        elif character == '+':
            index = 11
        elif character == '=':
            index = 12
        else:
            index = int(character)

        one_hot_encoding[index] = 1

        return one_hot_encoding

    def file_to_list_of_strings(self, file):
        list_of_x = []
        list_of_y = []
        for line in open(file, 'r'):
            stripped_line = line.strip()

            x,y = stripped_line.split(',')

            list_of_x.append(x.strip())
            list_of_y.append(y.strip())

        return list_of_x,list_of_y

    def training_size(self):
        return len(self.training_x)

    def testing_size(self):
        return len(self.testing_x)

    def training_item(self, key):
        idx = self.training_order[key]

        line = self.training_x[idx]

        X = []
        for char in line:
            X.append(self.character_to_vector(char))

        y = float(self.training_y[idx])

        return X, y, line

    def testing_item(self, key):
        idx = self.testing_order[key]

        line = self.testing_x[idx]

        X = []
        for char in line:
            X.append(self.character_to_vector(char))

        y = float(self.testing_y[idx])

        return X, y, line

    def __shuffle__(self, indices):
        return np.random.choice(indices,len(indices),replace=False)

    def shuffle_order(self):
        self.training_order = self.__shuffle__(self.training_order)
