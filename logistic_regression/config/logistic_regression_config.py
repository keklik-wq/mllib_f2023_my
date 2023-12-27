from easydict import EasyDict

from logistic_regression.utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization

# training
cfg.weights_init_type = WeightsInitType.xavier_normal
cfg.weights_init_kwargs = {'sigma': 1}

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 100

cfg.neptune_project = 'barabulka/dudkov-project'
cfg.neptune_token_path = "logistic_regression/config/token.env"
