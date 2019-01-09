from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import plt
from progressbar import ProgressBar

from models import sklearn_model


class DataUtils(object):
    def __init__(self, config):
        self.config = config

        self.origin_train_path = Path(self.config["datas"]["origin_train_path"])
        self.origin_test_path = Path(self.config["datas"]["origin_test_path"])
        self.generate_train_path = Path(self.config["datas"]["generate_train_path"])
        self.generate_validate_path = Path(self.config["datas"]["generate_validate_path"])
        self.generate_test_path = Path(self.config["datas"]["generate_test_path"])

        self.is_regenerate_train_data = self.config["datas"]["is_regenerate_train_data"]
        self.split_validate_size = self.config["datas"]["split_validate_size"]
        self.is_regenerate_test_data = self.config["datas"]["is_regenerate_test_data"]

        self.is_debug = self.config["is_debug"]
        self.is_std = self.config["is_std"]
        self.is_ntl = self.config["is_ntl"]
        self.is_remove_extreme = self.config["is_remove_extreme"]
        self.is_plot = 0

        self.progress = ProgressBar()

    def get_train_data(self):
        if self.is_regenerate_train_data:
            df_train_data = self.__generate_train_data()
        else:
            df_train_data = pd.read_csv(self.generate_train_path)

        if self.is_debug:
            print(f"\n>> train df head:\n\n{df_train_data.head()}")

        train_header_list = list(df_train_data.columns)
        train_remove_key_list = ["id", "date", "y"]
        for remove_key in train_remove_key_list:
            train_header_list.remove(remove_key)
        train_input_list = df_train_data[train_header_list].values
        train_target_list = df_train_data["y"].values

        return train_input_list, train_target_list

    def get_test_data(self):
        if self.is_regenerate_test_data:
            df_test_data = self.__generate_test_data()
        else:
            df_test_data = pd.read_csv(self.generate_test_path)

        if self.is_debug:
            print(f"\n>> test df head:\n\n{df_test_data.head()}")

        df_test_data = df_test_data.fillna(0)
        test_header_list = list(df_test_data.columns)
        train_remove_key_list = ["date", "id"]
        for remove_key in train_remove_key_list:
            test_header_list.remove(remove_key)
        test_input_list = df_test_data[test_header_list].values
        test_id_list = df_test_data["id"].values
        test_date_list = df_test_data["date"].values
        return test_input_list, test_id_list, test_date_list

    def __generate_train_data(self):
        sorted_train_path = sorted(self.origin_train_path.glob("*"))
        sorted_train_len = len(sorted_train_path)
        print(f">> all data length: {sorted_train_len}")

        if self.is_debug:
            sorted_train_path = sorted_train_path[:10]

        df_train_data = pd.DataFrame()
        for index, date_path in enumerate(self.progress(sorted_train_path)):
            df_train_data = pd.concat([df_train_data, self.__merge_date_data(date_path)], ignore_index=True)

        print(">> generate all train data success !")

        if not self.is_debug:
            df_train_data.to_csv(self.generate_train_path, index=False)
            print(">> save all data to /datas folder.")

        return df_train_data

    def __generate_test_data(self):
        df_test_data = pd.DataFrame()
        test_path_list = list(self.origin_test_path.glob("*"))
        for date_path in self.progress(test_path_list):
            df_test_data = pd.concat([df_test_data, self.__merge_date_data(date_path, "test")])

        print(">> save all data to /datas folder.")
        df_test_data.to_csv(self.generate_test_path)
        print(">> generate test data success !")

        return df_test_data

    def __merge_date_data(self, date_path, data_type=None):
        df_non_ts = pd.read_csv(date_path / "non_ts.csv", index_col="id")
        if self.is_remove_extreme:
            df_non_ts = self.remove_extreme_value(df_non_ts)
        if self.is_ntl:
            df_non_ts = self.neutralization(df_non_ts)
        if self.is_std:
            df_non_ts = self.standardization(df_non_ts)

        if data_type == "test":
            merge_df = df_non_ts
        else:
            df_y = pd.read_csv(date_path / "y.csv")
            del df_y["date"]
            merge_df = pd.merge(df_non_ts, df_y, on="id")

        for ts_path in date_path.glob("ts_*.csv"):
            ts_name = ts_path.name.split(".csv")[0]
            df_ts = pd.read_csv(ts_path, index_col="id")
            date = df_ts["date"]
            del df_ts["date"]

            if self.is_remove_extreme:
                df_ts = self.remove_extreme_value(df_ts)
            if self.is_std:
                df_ts = self.standardization(df_ts)

            df_ts_std = self.__get_std(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_std, on="id")

            df_ts_mean_0_5 = self.__get_mean_0_5(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_5, on="id")

            df_ts_mean_0_20 = self.__get_mean_0_20(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_20, on="id")

            merge_df["date"] = date

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

    def remove_extreme_value(self, df_input):
        if self.is_plot:
            del df_input["date"]
            del df_input["flag"]

            fig, (ax0, ax1) = plt.subplots(2, 1, sharey="all")
            ax0.set_title('BEFORE /20130201/non_ts.csv remove extreme value')
            df_input.plot(ax=ax0)

            desc = df_input.describe()
            mean_add_3std = desc.loc['mean'] + desc.loc['std'] * 3
            mean_minus_3std = desc.loc['mean'] - desc.loc['std'] * 3
            df_input = df_input.where(df_input < mean_add_3std, mean_add_3std, axis=1)
            df_input = df_input.where(df_input > mean_minus_3std, mean_minus_3std, axis=1)

            ax1.set_title('AFTER /20130201/non_ts.csv remove extreme value')
            df_input.plot(ax=ax1)
            plt.show()
        else:
            desc = df_input.describe()
            mean_add_3std = desc.loc['mean'] + desc.loc['std'] * 3
            mean_minus_3std = desc.loc['mean'] - desc.loc['std'] * 3
            df_input = df_input.where(df_input < mean_add_3std, mean_add_3std, axis=1)
            df_input = df_input.where(df_input > mean_minus_3std, mean_minus_3std, axis=1)

        return df_input

    # 标准化
    @staticmethod
    def standardization(df_input):
        df_input = df_input.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        df_input = df_input.fillna(0)
        return df_input

    # 中性化
    @staticmethod
    def neutralization(df_input):
        flag_values = [[v] for v in df_input["flag"].values]
        cols = list(df_input.columns)
        model = sklearn_model.Model()
        skm = model.sklearn_model("LinearRegression")

        cols.remove("date")
        cols.remove("flag")
        for col in cols:
            col_values = [[v] for v in df_input[col].values]
            skm.fit(flag_values, col_values)
            col_preds = skm.predict(flag_values)
            res = []
            for x, y in zip(col_values, col_preds):
                res.append(x[0] - y[0])
            df_input[col] = res

        return df_input
