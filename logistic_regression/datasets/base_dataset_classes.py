from abc import ABC, abstractmethod

import numpy as np


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

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

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        number_of_rows = len(self.inputs)
        len_of_train = int(number_of_rows * self.train_set_percent)
        len_of_valid = int(number_of_rows * self.valid_set_percent)

        indexes = np.random.permutation(number_of_rows)

        inputs = self.inputs[indexes]
        targets = self.targets[indexes]

        self.inputs_train = inputs[:len_of_train]
        self.target_train = targets[:len_of_train]

        self.inputs_test = inputs[len_of_train:len_of_train + len_of_valid]
        self.target_test = targets[len_of_train:len_of_train + len_of_valid]

        self.inputs_valid = inputs[len_of_train + len_of_valid:]
        self.target_valid = targets[len_of_train + len_of_valid:]

    def normalization(self):
        # TODO write normalization method BONUS TASK
        min_inputs = np.min(self.inputs, 0)
        max_inputs = np.max(self.inputs, 0)
        lambda_function = lambda x: 2 * (x - min_inputs) / (max_inputs - min_inputs) - 1
        self.input = np.vectorize(lambda_function)(self.inputs)

    def get_data_stats(self):
        self.mean = np.mean(self.inputs, axis=0)
        self.std = np.std(self.inputs, axis=0)
        # TODO calculate mean and std of inputs vectors of training set by each dimension

    def standartization(self):
        lambda_function = lambda x: (x - self.mean) / self.std
        self.input = np.vectorize(lambda_function)(self.inputs)

        # TODO write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        return np.eye(number_classes)[targets]
        # TODO create matrix of onehot encoding vactors for input targets
        # it is possible to do it without loop try make it without loop

