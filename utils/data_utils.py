from pathlib import Path

import pandas as pd

from utils import config_utils


config_path = "./config.yaml"
config = config_utils.read_config(config_path)


def merge_date_data():
    train_path = Path(config["datas"]["train_path"])
    for date_path in train_path.glob("*"):
        df_non_ts = pd.read_csv(date_path / "non_ts.csv", index_col="id")
        df_y = pd.read_csv(date_path / "y.csv")
        merge_df = pd.merge(df_non_ts, df_y, on="id")

        for ts_path in date_path.glob("ts_*.csv"):
            ts_name = ts_path.name.split(".csv")[0]
            df_ts = pd.read_csv(ts_path, index_col="id")

            df_ts_std = get_std(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_std, on="id")

            df_ts_mean_0_5 = get_mean_0_5(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_5, on="id")

            df_ts_mean_0_20 = get_mean_0_20(df_ts, ts_name)
            merge_df = pd.merge(merge_df, df_ts_mean_0_20, on="id")

        print(merge_df.head())
        break


def get_std(df_input, index_name):
    df_output = df_input.T[1:].std().to_frame(name=index_name+"_std")
    return df_output


def get_mean_0_5(df_input, index_name):
    df_output = df_input.T[1:7].mean().to_frame(name=index_name+"_mean_0_5")
    return df_output


def get_mean_0_20(df_input, index_name):
    df_output = df_input.T[1:22].mean().to_frame(name=index_name+"_mean_0_20")
    return df_output


if __name__ == '__main__':
    merge_date_data()
