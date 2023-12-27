#
# инициализация класса набора данных, стандартизация данных, разделение на выборки, построение onehot encoding вектора
# инициализация класса логистической регрессии
# обучение модели, логирование в Нептун
# сохранение модели

from datasets.base_dataset_classes import BaseClassificationDataset
from datasets.digits_dataset import Digits
from config.logistic_regression_config import cfg
from models.logistic_regression_model import LogReg

data = Digits(cfg)
model = LogReg(cfg, data.k, data.d, "Logistic regression")
model.weights_init_xavier_normal()
model.train(
    data.inputs_train, BaseClassificationDataset.onehotencoding(data.target_train, data.k),
    data.inputs_valid, BaseClassificationDataset.onehotencoding(data.target_valid, data.k)
)
