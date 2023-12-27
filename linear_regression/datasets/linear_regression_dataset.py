import numpy as np
from linear_regression.utils.common_functions import read_dataframe_file
from easydict import EasyDict
from linear_regression.datasets.base_dataset import BaseDataset
from linear_regression.utils.enums import SetType


class LinRegDataset(BaseDataset):

    def __init__(self, cfg: EasyDict, inputs_cols='inputs', target_cols='targets'):
        super(LinRegDataset, self).__init__(cfg.train_set_percent, cfg.valid_set_percent, cfg.is_shuffled)

        advertising_dataframe = read_dataframe_file(cfg.dataframe_path)

        # define properties
        self.inputs = np.asarray(advertising_dataframe[inputs_cols])
        self.targets = np.asarray(advertising_dataframe[target_cols])

        # divide into sets
        self._divide_into_sets()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if isinstance(value, np.ndarray):
            self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    def __call__(self, set_type: SetType) -> dict:
        return {'inputs': getattr(self, f'inputs_{set_type.name}'),
                'targets': getattr(self, f'targets_{set_type.name}'), }


if __name__ == '__main__':
    from linear_regression.configs.linear_regression_cfg import cfg

    lin_reg_dataset1 = LinRegDataset(cfg, inputs_cols=['x_0', 'x_1', 'x_2'], target_cols=['y_0'])
    lin_reg_dataset = LinRegDataset(cfg)
    print(np.size(lin_reg_dataset.inputs_train))
    print(np.size(lin_reg_dataset.inputs_test))
    print(np.size(lin_reg_dataset.inputs_valid))
