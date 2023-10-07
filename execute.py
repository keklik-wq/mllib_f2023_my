# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import pandas as pd

import models.linear_regression_model as lr_model
import datasets.linear_regression_dataset as ds
import configs.linear_regression_cfg as cf
from utils.enums import TrainType
import utils.visualisation as vis
import utils.metrics as metric


data = ds.LinRegDataset(cf.cfg)

cf.cfg.train_type = TrainType.gradient_descent

model_1 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)
model_1.train(data.inputs_train, data.target_train)
prediction = model_1.__call__(data.inputs_valid)

graph = vis.Visualisation()
graph.compare_model_predictions(data.inputs_valid, prediction, data.target_valid, "valid")

"""
cf.cfg.train_type = TrainType.gradient_descent
cf.cfg.degree = 8
model_2 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)

cf.cfg.train_type = TrainType.gradient_descent
cf.cfg.degree = 100
model_3 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)

cf.cfg.train_type = TrainType.normal_equation
cf.cfg.degree = 1
model_4 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)

cf.cfg.train_type = TrainType.normal_equation
cf.cfg.degree = 8
model_5 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)

cf.cfg.train_type = TrainType.normal_equation
cf.cfg.degree = 100
model_6 = lr_model.LinearRegression(cf.cfg.base_functions, 0.01)
"""