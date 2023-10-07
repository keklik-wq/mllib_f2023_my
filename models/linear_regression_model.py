import sys

import numpy as np
from configs.linear_regression_cfg import cfg
from utils.enums import TrainType
from logs.Logger import Logger
import cloudpickle

from utils.metrics import mse


class LinearRegression():

    def __init__(self,
                 base_functions: list,
                 learning_rate: float,
                 reg_coefficient: float = 0.0,
                 experiment_name: str = "experiment_1"):
        self.weights = np.random.randn(len(base_functions) + 1)
        self.base_functions = base_functions
        self.learning_rate = learning_rate
        self.reg_coefficient = reg_coefficient
        # self.neptune_logger = Logger(cfg.env_path, cfg.project_name, experiment_name)

    # Methods related to the Normal Equation

    def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        u, s, v = np.linalg.svd(matrix)
        epsilon = sys.float_info.epsilon

        n = matrix.shape[0]
        m = matrix.shape[1] - np.count_nonzero(self.base_functions.transpose()[0] == 1)

        max_s = np.max(s)
        max_value = epsilon * max(n, m + 1) * max_s

        s_pseudo = np.where(s > max_value, 1 / s, 0)
        s_pseudo = np.transpose(s_pseudo)

        pseudo_inverse = np.matmul(v * s_pseudo, np.transpose(u))
        return pseudo_inverse

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        self.weights = np.matmul(pseudoinverse_plan_matrix, targets)

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:

        plan_matrix = np.array(
            [np.vectorize(self.base_functions[i])(inputs) for i in range(np.size(self.base_functions))]).transpose()

        plan_matrix_with_ones = np.insert(plan_matrix, 0, 1, axis=1)

        print(plan_matrix_with_ones)
        return plan_matrix_with_ones

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:

        weights_t = self.weights.T

        predictions = np.dot(plan_matrix, weights_t)

        return predictions

    def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:

        N = plan_matrix.shape[0]

        predictions = np.dot(plan_matrix, self.weights)

        error = predictions - targets

        gradient = (2 / N) * np.dot(np.transpose(plan_matrix), error)

        return gradient

    def calculate_cost_function(self, plan_matrix, targets):
        predictions = np.dot(plan_matrix, self.weights)

        squared_error = mse(predictions, targets)

        return squared_error

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Train the model using either the normal equation or gradient descent based on the configuration.
        TODO: Complete the training process.
        """
        plan_matrix = self._plan_matrix(inputs)
        if cfg.train_type.value == TrainType.normal_equation.value:
            pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix)
            # train process
            self._calculate_weights(pseudoinverse_plan_matrix, targets)
        else:
            """
            At each iteration of gradient descent, the weights are updated using the formula:
        
            w_{k+1} = w_k - γ * ∇_w E(w_k)
        
            Where:
            - w_k is the current weight vector at iteration k.
            - γ is the learning rate, determining the step size in the direction of the negative gradient.
            - ∇_w E(w_k) is the gradient of the cost function E with respect to the weights w at iteration k.
        
            This iterative process aims to find the weights that minimize the cost function E(w).
        """
            for current_epoch in cfg.epoch:
                gradient = self._calculate_gradient(plan_matrix, targets)
                self.weights = self.weights - self.learning_rate * gradient
                # update weights w_{k+1} = w_k - γ * ∇_w E(w_k)

                if current_epoch % 10 == 0:
                    print(self.calculate_cost_function(plan_matrix, targets))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
