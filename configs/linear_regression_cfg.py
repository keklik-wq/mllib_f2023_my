from easydict import EasyDict
import numpy as np
cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = '../linear_regression_dataset.csv'

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.is_shuffled = True




