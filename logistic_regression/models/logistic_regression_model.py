from typing import Union
import numpy as np
from easydict import EasyDict
from logistic_regression.logs.Logger import Logger


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int, experiment_name: str):
        self.bies = None
        self.weights = None
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        self.neptune_logger = Logger(cfg.neptune_token_path, cfg.neptune_project, experiment_name)
        self.eps = None
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')()  # **cfg.weights_init_kwargs)

    def weights_init_xavier_normal(self,loc = 0):
        scale = np.sqrt(2 / (self.d + self.k))

        self.weights = np.random.normal(loc=loc, scale=scale, size=(self.k, self.d))
        self.bies = np.random.normal(loc=loc, scale=scale, size=self.k)
        #  init weights with Xavier normal W ~ N(0, sqrt(2 / (D + K)))


    def weights_init_xavier_uniform(self):
        value = np.sqrt(6 / (self.d + self.k))
        self.weights = np.random.uniform(-value, value, size=(self.k, self.d))
        self.bies = np.random.uniform(-value, value, size=self.k)

        # init weights with Xavier uniform W ~ U(-sqrt(6 / (D + K)), sqrt(6 / (D + K)))

    def weights_init_he_normal(self,loc = 0):
        scale = np.sqrt(2 / (self.d))
        # init weights with He normal W ~ N(0, sqrt(2 / D))

        self.weights = np.random.normal(loc=loc, scale=scale, size=(self.k, self.d))
        self.bies = np.random.normal(loc=loc, scale=scale, size=self.k)

    def weights_init_he_uniform(self):
        value = np.sqrt(6 / (self.d))
        self.weights = np.random.uniform(-value, value, size=(self.k, self.d))
        self.bies = np.random.uniform(-value, value, size=self.k)
        #  init weights with He uniform W ~ U(-sqrt(6 / D), sqrt(6 / D)

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        """
        Computes the softmax function on the model output.

               The formula for softmax function is:
               y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

               where:
               - y_j is the softmax probability of class j,
               - z_j is the model output for class j before softmax,
               - K is the total number of classes,
               - Σ denotes summation.

               For numerical stability, subtract the max value of model_output before exponentiation:
               z_j = z_j - max(model_output)

               Parameters:
               model_output (np.ndarray): The model output before softmax.

               Returns:
               np.ndarray: The softmax probabilities.
             implement this function
        :param model_output:
        :return: softmax of output :)
        """
        result = np.exp((model_output - np.max(model_output, axis=1)[:, np.newaxis]))
        return result / result.sum(axis=1)[:, np.newaxis]


    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """
                Calculates model confidence using the formula:
                y(x, b, W) = Softmax(Wx + b) = Softmax(z)

                Parameters:
                inputs (np.ndarray): The input data.

                Returns:
                np.ndarray: The model confidence.
        """
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        """
        This function computes the model output by applying a linear transformation
        to the input data.

        The linear transformation is defined by the equation:
        z = W * x + b

        where:
        - W (a KxD matrix) represents the weight matrix,
        - x (a DxN matrix, also known as 'inputs') represents the input data,
        - b (a vector of length K) represents the bias vector,
        - z represents the model output before activation.

        Returns:
        np.ndarray: The model output before softmax.

        TODO implement this function  using matrix multiplication DO NOT USE LOOPS
        """
        z = lambda x: self.weights @ x + self.bies
        return np.apply_along_axis(z, axis=1, arr=inputs)

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        """
        temp_arr = np.concatenate([model_confidence - targets, inputs], axis=1)
        d = lambda x: np.outer(x[:self.k], x[self.k:].T)
        res = np.sum(
            np.apply_along_axis(d, arr=temp_arr, axis=1),
            axis=0
        )
        return res
        :param inputs:
        :param targets:
        :param model_confidence:
        :return:
        """
        temp_arr = np.concatenate([model_confidence - targets, inputs], axis=1)
        outer_product = np.matmul((temp_arr[:, :self.k]).reshape(-1, self.k, 1),
                                  (temp_arr[:, self.k:]).reshape(-1, 1, temp_arr.shape[1] - self.k))
        return np.sum(outer_product, axis=0)

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # calculate gradient for b
        return np.sum(model_confidence - targets, 0)

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        #  update model weights
        self.weights -= self.cfg.gamma * self.__get_gradient_w(inputs, targets, model_confidence)
        self.bies -= self.cfg.gamma * self.__get_gradient_b(targets, model_confidence)

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None,
                                brevity_of_logging: int = 5):
        #  one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #
        #  update weights
        #   you can add some other steps if you need
        # log your results in Neptune
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """
        train_target_fun = self.__target_function_value(inputs_train, targets_train)

        metrics_valid = self.__validate(inputs_valid, targets_valid)
        metrics_train = self.__validate(inputs_train, targets_train)

        self.__weights_update(inputs_train,
                              targets_train,
                              self.get_model_confidence(inputs_train)
                              )

        self.neptune_logger.log_hyperparameters(
            {
                "TRAIN.target_function": train_target_fun,
                "TRAIN.accuracy": metrics_train["accuracy"]
            }
        )
        if epoch % brevity_of_logging == 0:
            valid_target_fun = self.__target_function_value(inputs_valid, targets_valid)
            self.neptune_logger.log_hyperparameters(
                {
                    "VALID.target_function": valid_target_fun,
                    "VALID.accuracy": metrics_valid["accuracy"]
                }
            )

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        epoch = 0

        while np.linalg.norm(self.__get_gradient_w(inputs_train,
                                                   targets_train,
                                                   self.get_model_confidence(inputs_train)
                                                   )
                            ) >= self.eps:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            epoch += 1

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        epoch = 0
        weights_before_update = self.weights
        self.__weights_update(inputs_train, targets_train, self.get_model_confidence(inputs_train))
        weights_after_update = self.weights

        while np.linalg.norm(weights_before_update - weights_after_update) >= self.eps:
            self.__gradient_descent_step(inputs_train,
                                         targets_train,
                                         epoch,
                                         inputs_valid,
                                         targets_valid
                                         )
            epoch += 1

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        # gradient_descent with stopping criteria - metric (accuracy, f1 score, or other) value on the validation set is not improving
        # Here, we are performing gradient descent until the validation accuracy stops improving by more than epsilon
        epoch = 0

        while self.__validate(inputs_train, targets_train)['accuracy'] >= self.eps:
            self.__gradient_descent_step(inputs_train,
                                         targets_train,
                                         epoch,
                                         inputs_valid,
                                         targets_valid
                                         )
            epoch += 1

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid, targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                z: Union[np.ndarray, None] = None) -> float:
        """
        This function computes the target function value based on the formula:

        Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^(z_il)) - z_ik)
        where:
        - N is the size of the data set,
        - K is the number of classes,
        - t_{ik} is the target value for data point i and class k,
        - z_{il} is the model output before softmax for data point i and class l,
        - z is the model output before softmax (matrix z).

        Parameters:
        inputs (np.ndarray): The input data.
        targets (np.ndarray): The target data.
        z (Union[np.ndarray, None]): The model output before softmax. If None, it will be computed.

        Returns:
        float: The value of the target function.
        implement this function
        """
        if z is None:
            z = self.__get_model_output(inputs)

        size_of_model_output = np.shape(z)
        max_mo = np.max(z, axis=1)
        max_mo = np.tile(max_mo, (size_of_model_output[1], 1)).T
        z -= max_mo

        res = np.sum(targets * (np.tile(
            np.log(
                np.sum(
                    np.exp(z),
                    axis=1
                )
            ),
            (self.k, 1)
        ).T - z))
        return res

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        #  metrics calculation: accuracy, confusion matrix
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)

        size_of_inputs = np.shape(inputs)[0]
        predictions = np.argmax(model_confidence, axis=1)
        confusion_matrix = np.zeros((self.k, self.k))
        targets = np.argmax(targets, axis=1)

        accuracy = np.sum(predictions == targets) / size_of_inputs
        for i in range(size_of_inputs):
            confusion_matrix[predictions[i], targets[i]] += 1

        return {"accuracy": accuracy, "confusion matrix": confusion_matrix}

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions
