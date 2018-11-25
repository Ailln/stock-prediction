from pathlib import Path

import pandas as pd

from utils import config_utils

is_debug = True


class DataUtils(object):
    def __init__(self, config_path):
        self.config = config_utils.read_config(config_path)

        self.origin_train_path = Path(self.config["datas"]["origin_train_path"])
        self.origin_test_path = Path(self.config["datas"]["origin_test_path"])
        self.generate_train_path = Path(self.config["datas"]["generate_train_path"])
        self.generate_validate_path = Path(self.config["datas"]["generate_validate_path"])
        self.generate_test_path = Path(self.config["datas"]["generate_test_path"])

        self.is_regenerate_train_and_validate_data = self.config["datas"]["is_regenerate_train_and_validate_data"]
        self.split_validate_size = self.config["datas"]["split_validate_size"]

    def generate_train_and_validate_data(self):
        if is_debug:
            self.is_regenerate_train_and_validate_data = 1

        if self.is_regenerate_train_and_validate_data:
            sorted_all_data_path = sorted(self.origin_train_path.glob("*"))
            sorted_all_data_len = len(sorted_all_data_path)
            print(f">> all data length: {sorted_all_data_len}")

            df_train_data = pd.DataFrame()
            df_validate_data = pd.DataFrame()
            for index, date_path in enumerate(sorted_all_data_path):
                if index <= sorted_all_data_len * (1 - self.split_validate_size):
                    df_train_data = pd.concat([df_train_data, self.__merge_date_data(date_path)], ignore_index=True)
                else:
                    df_validate_data = pd.concat([df_validate_data, self.__merge_date_data(date_path)],
                                                 ignore_index=True)

                if not index % 10:
                    print(f">> {index}")

            df_train_data.to_csv(self.generate_train_path)
            df_validate_data.to_csv(self.generate_validate_path)
            print(">> generate train and validate data success !")

    def __merge_date_data(self, date_path):
        df_non_ts = pd.read_csv(date_path / "non_ts.csv", index_col="id")
        df_y = pd.read_csv(date_path / "y.csv")
        merge_df = pd.merge(df_non_ts, df_y, on="id")

        for ts_path in date_path.glob("ts_*.csv"):
            ts_name = ts_path.name.split(".csv")[0]
            df_ts = pd.read_csv(ts_path, index_col="id")

            df_ts_std = self.__get_std(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_std, on="id")

            df_ts_mean_0_5 = self.__get_mean_0_5(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_5, on="id")

            df_ts_mean_0_20 = self.__get_mean_0_20(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_20, on="id")

        return merge_df

    @staticmethod
    def __get_std(df_input, index_name):
        df_output = df_input.T[1:].std().to_frame(name=index_name+"_std")
        return df_output

    @staticmethod
    def __get_mean_0_5(df_input, index_name):
        df_output = df_input.T[1:7].mean().to_frame(name=index_name+"_mean_0_5")
        return df_output

    @staticmethod
    def __get_mean_0_20(df_input, index_name):
        df_output = df_input.T[1:22].mean().to_frame(name=index_name+"_mean_0_20")
        return df_output


if __name__ == '__main__':
    my_config_path = "./config.yaml"
    du = DataUtils(my_config_path)
    du.generate_train_and_validate_data()
