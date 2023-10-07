from easydict import EasyDict
from utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = 'linear_regression_dataset.csv'


# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.

cfg.number_of_polynoms = 3

cfg.base_functions = [lambda x, i=i: x ** i for i in range(1, cfg.number_of_polynoms+1)]


cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.is_shuffled = True

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = range(100)

#cfg.exp_name = ''
cfg.env_path = '' # Путь до файла .env где будет храниться api_token.
cfg.project_name = ''

