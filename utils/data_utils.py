from pathlib import Path

import pandas as pd

from utils import config_utils


config_path = "./config.yaml"
config = config_utils.read_config(config_path)


def merge_date_data():
    train_path = Path(config["datas"]["train_path"])
    for date_path in train_path.glob("*"):
        df_non_ts = pd.read_csv(date_path / "non_ts.csv")
        df_ts_1 = pd.read_csv(date_path / "ts_1.csv")
        df_ts_2 = pd.read_csv(date_path / "ts_2.csv")
        df_ts_3 = pd.read_csv(date_path / "ts_3.csv")
        df_ts_4 = pd.read_csv(date_path / "ts_4.csv")
        df_ts_5 = pd.read_csv(date_path / "ts_5.csv")
        df_y = pd.read_csv(date_path / "y.csv")
        merge_df = pd.merge(df_non_ts, df_ts_1, on="id")
        print(merge_df.head())
        break


if __name__ == '__main__':
    merge_date_data()
