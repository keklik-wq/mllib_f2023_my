from easydict import EasyDict
from linear_regression.utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = 'linear_regression_dataset.csv'

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.is_shuffled = True

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = range(10)

#cfg.exp_name = ''
cfg.env_path = '' # Путь до файла .env где будет храниться api_token.
cfg.project_name = ''

