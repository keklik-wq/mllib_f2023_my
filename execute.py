# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import random

import pandas as pd
from numpy import sin, cos, log, exp

import models.linear_regression_model as lr_model
import datasets.linear_regression_dataset as ds
import configs.linear_regression_cfg as cf
from utils.enums import TrainType
import utils.visualisation as vis
import utils.metrics as metric

data = ds.LinRegDataset(cf.cfg)

cf.cfg.train_type = TrainType.gradient_descent


def train_model(number_of_polynom, learning_rate, reg_coefficient, number_of_function):
    base_functions = dict([("polynom", [lambda x, i=i: x ** i for i in range(1, number_of_polynom + 1)]),
                           ("sin", [lambda x: sin(x)]),
                           ("cos", [lambda x: cos(x)]),
                            ("exp",[lambda x: exp(x)])
                      ])
    model_1 = lr_model.LinearRegression(base_functions[number_of_function], learning_rate, reg_coefficient)
    model_1.train(data.inputs_train, data.target_train)
    prediction = model_1.__call__(data.inputs_valid)
    print(metric.mse(prediction, data.target_valid))
    return prediction


m_min = 1
m_max = 100

rc_min = 0.001
rc_max = 0.1

lr_min = 0.1
lr_max = 0.2

predictions = []

for _ in range(0, 30):
    number_of_polynoms = random.randint(m_min, m_max)
    regularizaton = random.uniform(rc_min, rc_max)
    learning_rate = random.uniform(lr_min, lr_max)
    number_of_function = random.choice(["polynom","sin","cos","exp"])
    print(number_of_polynoms, regularizaton, learning_rate, number_of_function)
    regularizaton = random.uniform(lr_min,lr_max)
    predictions.append(
        [
            train_model(number_of_polynom=number_of_polynoms,
                        reg_coefficient=regularizaton,
                        learning_rate=learning_rate,
                        number_of_function=number_of_function),
            dict(
                [
                    ("number of polynoms:", number_of_polynoms),
                    ("regularization", regularizaton),
                    ("learning_rate", learning_rate),
                    ("function", number_of_function)
                ]
            )
        ]
    )

graph = vis.Visualisation()
graph.compare_model_predictions(data.inputs_valid, predictions, data.target_valid, "valid")
