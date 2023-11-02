import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent, is_shuffled = True):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent
        self.is_shuffled = is_shuffled

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass


    def _divide_into_sets(self):
        number_of_rows = len(self.inputs)
        len_of_train = int(number_of_rows * self.train_set_percent)
        len_of_valid = int(number_of_rows * self.valid_set_percent)

        if self.is_shuffled:
            indexes = np.random.permutation(number_of_rows)
        else:
            indexes = np.arange(1, number_of_rows)

        inputs = self.inputs[indexes]
        targets = self.targets[indexes]

        self.inputs_train = inputs[:len_of_train]
        self.target_train = targets[:len_of_train]

        self.inputs_test = inputs[len_of_train:len_of_train + len_of_valid]
        self.target_test = targets[len_of_train:len_of_train + len_of_valid]

        self.inputs_valid = inputs[len_of_train + len_of_valid:]
        self.target_valid = targets[len_of_train + len_of_valid:]
