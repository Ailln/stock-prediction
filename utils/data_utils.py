from pathlib import Path

import pandas as pd
from progressbar import ProgressBar


class DataUtils(object):
    def __init__(self, config):
        self.config = config

        self.origin_train_path = Path(self.config["datas"]["origin_train_path"])
        self.origin_test_path = Path(self.config["datas"]["origin_test_path"])
        self.generate_train_path = Path(self.config["datas"]["generate_train_path"])
        self.generate_validate_path = Path(self.config["datas"]["generate_validate_path"])
        self.generate_test_path = Path(self.config["datas"]["generate_test_path"])

        self.is_regenerate_train_and_validate_data = self.config["datas"]["is_regenerate_train_and_validate_data"]
        self.split_validate_size = self.config["datas"]["split_validate_size"]

        self.is_debug = self.config["is_debug"]

        self.progress = ProgressBar()

    def get_train_and_validate_data(self):
        if self.is_regenerate_train_and_validate_data:
            df_train_data, df_validate_data = self.__generate_train_and_validate_data()
        else:
            df_train_data = pd.read_csv(self.generate_train_path)
            df_validate_data = pd.read_csv(self.generate_validate_path)

        if self.is_debug:
            print(f"\n>> train df head:\n\n{df_train_data.head()}")
            print(f"\n>> validate df head:\n\n{df_validate_data.head()}")

        train_header_list = list(df_train_data.columns)
        train_remove_key_list = ["id", "date", "y"]
        for remove_key in train_remove_key_list:
            train_header_list.remove(remove_key)
        train_input_list = df_train_data[train_header_list].values
        train_target_list = df_train_data["y"].values

        validate_header_list = list(df_validate_data.columns)
        validate_remove_key_list = ["id", "date", "y"]
        for remove_key in validate_remove_key_list:
            validate_header_list.remove(remove_key)
        validate_input_list = df_validate_data[validate_header_list].values
        validate_target_list = df_validate_data["y"].values

        return train_input_list, train_target_list, validate_input_list, validate_target_list

    def __generate_train_and_validate_data(self):
        sorted_all_data_path = sorted(self.origin_train_path.glob("*"))
        sorted_all_data_len = len(sorted_all_data_path)
        print(f">> all data length: {sorted_all_data_len}")

        if self.is_debug:
            sorted_all_data_path = sorted_all_data_path[:10]
            sorted_all_data_len = 10

        df_train_data = pd.DataFrame()
        df_validate_data = pd.DataFrame()
        print(">> concat all data")
        for index, date_path in enumerate(self.progress(sorted_all_data_path)):
            if index <= sorted_all_data_len * (1 - self.split_validate_size):
                df_train_data = pd.concat([df_train_data, self.__merge_date_data(date_path)], ignore_index=True)
            else:
                df_validate_data = pd.concat([df_validate_data, self.__merge_date_data(date_path)],
                                             ignore_index=True)

        print(">> save all data to /datas folder.")
        df_train_data.to_csv(self.generate_train_path, index=False)
        df_validate_data.to_csv(self.generate_validate_path, index=False)
        print(">> generate train and validate data success !")

        return df_train_data, df_validate_data

    def __merge_date_data(self, date_path):
        df_non_ts = pd.read_csv(date_path / "non_ts.csv", index_col="id")
        df_y = pd.read_csv(date_path / "y.csv")
        del df_y["date"]
        merge_df = pd.merge(df_non_ts, df_y, on="id")

        for ts_path in date_path.glob("ts_*.csv"):
            ts_name = ts_path.name.split(".csv")[0]
            df_ts = pd.read_csv(ts_path, index_col="id")
            del df_ts["date"]

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
    du.get_train_and_validate_data()
