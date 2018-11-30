import unittest

import pandas as pd

from utils import config_utils
from utils import data_utils


class TestDataUtils(unittest.TestCase):
    def setUp(self):
        self.config = config_utils.read_config("./config.yaml")
        self.test_df = pd.read_csv(self.config["datas"]["origin_train_path"] + "/20130201/non_ts.csv", index_col="id")
        self.du = data_utils.DataUtils(self.config)

    def tearDown(self):
        pass

    # def test_remove_extreme_value(self):
    #     self.du.remove_extreme_value(self.test_df)

    def test_standardization(self):
        self.du.standardization(self.test_df)


if __name__ == '__main__':
    unittest.main()
