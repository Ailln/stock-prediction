import os
import datetime
import argparse

import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from utils import data_utils
from utils import other_utils
from utils import config_utils
from models import sklearn_model

now = datetime.datetime.now()
print(f">> Start: {now}")

config_path = "./config.yaml"
config = config_utils.read_config(config_path)
default_model_name = config["models"]["model_name"]
all_model_name = config["models"]["all_model_name"]
save_model_path = config["models"]["model_path"]
log_model_path = config["models"]["log_path"]
split_validate_size = config["datas"]["split_validate_size"]
default_standardization = config["datas"]["is_standardization"]
default_neutralization = config["datas"]["is_neutralization"]
default_remove_extreme = config["datas"]["is_remove_extreme"]
default_cross_validate = config["datas"]["is_cross_validate"]
default_split_validate = config["datas"]["is_split_validate"]

parser = argparse.ArgumentParser()
parser.add_argument("--run_type", type=str, default="train", choices=["train", "test"])
parser.add_argument("--model_name", type=str, default=default_model_name, choices=all_model_name)
parser.add_argument("--train_type", type=str, default="single", choices=["single", "all"])
parser.add_argument("--is_std", type=other_utils.bool, default=default_standardization, choices=[True, False])
parser.add_argument("--is_ntl", type=other_utils.bool, default=default_neutralization, choices=[True, False])
parser.add_argument("--is_remove_extreme", type=other_utils.bool, default=default_remove_extreme, choices=[True, False])
parser.add_argument("--is_cross_val", type=other_utils.bool, default=default_cross_validate, choices=[True, False])
parser.add_argument("--is_split_val", type=other_utils.bool, default=default_split_validate, choices=[True, False])
parser.add_argument("--debug", type=other_utils.bool, default=False, choices=[True, False])
args = parser.parse_args()
print(f">> Parameters: {args}")
with open(f"{log_model_path}/{now}.log", "a") as f_log:
    f_log.write(str(args))

config["is_debug"] = args.debug
config["is_std"] = args.is_std
config["is_ntl"] = args.is_ntl
config["is_remove_extreme"] = args.is_remove_extreme
du = data_utils.DataUtils(config)


def train():
    print(">> data processing...")
    all_train_input, all_train_target = du.get_train_data()

    print(">> build model...")
    model = sklearn_model.Model()

    if args.train_type == "single":
        model_name = args.model_name
        print(f">> run {model_name} model...")

        train_utils(model, model_name, all_train_input, all_train_target)

    elif args.train_type == "all":
        # TODO 使用多线程加速
        for model_name in all_model_name:
            train_utils(model, model_name, all_train_input, all_train_target)

    else:
        raise ValueError(args.train_type)


def train_utils(model, model_name, all_train_input, all_train_target):
    skm = model.sklearn_model(model_name)
    # 是否切分出验证集
    # 在选择出模型后，需要使用训练集对模型进行全量训练
    # 此时要保存模型，以供预测时使用
    if args.is_split_val:
        # 是否使用交叉验证
        if args.is_cross_val:
            r2_result = cross_val_score(skm, all_train_input, all_train_target, cv=3, scoring='r2', n_jobs=-1)
            log = f"\n>> {model_name}\n>> ALL Cross R2: {r2_result}\n>> Mean R2: {r2_result.mean()}\n"
        else:
            train_input, validate_input, train_target, validate_target = \
                train_test_split(all_train_input, all_train_target, test_size=split_validate_size)
            skm.fit(train_input, train_target)
            validate_preds = skm.predict(validate_input)
            r2_result = r2_score(validate_target, validate_preds)
            log = f"\n>> {model_name}\n>> R2: {r2_result}"

        print(log)
        with open(f"{log_model_path}/{now}.log", "a") as f_log:
            f_log.write(log)

    else:
        skm.fit(all_train_input, all_train_target)
        save_path = f"{save_model_path}/{model_name}.m"
        print(f">> save model in {save_path}...")
        joblib.dump(model, save_path)


def test():
    print(">> data processing...")
    test_input, test_id, test_date = du.get_test_data()

    model_name = args.model_name
    save_path = f"{save_model_path}/{model_name}.m"
    print(f">> read model from {save_path}...")
    model = joblib.load(save_path)
    skm = model.sklearn_model(model_name)

    print(">> predict...\n")
    test_preds = skm.predict(test_input)

    result = {}
    date_set = set(test_date)
    for date in date_set:
        print(date)
        save_dir = f"./save/test/{str(date)}"
        save_path = save_dir + "/y.csv"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for date_item, id_item, value_item in zip(test_date, test_id, test_preds):
            if date_item == date:
                if date_item in result:
                    result[date_item].append([date_item, id_item, value_item])
                else:
                    result[date_item] = [[date_item, id_item, value_item]]

        sort_result = sorted(result[date], key=lambda x: x[2])
        sort_result.reverse()
        df_save = pd.DataFrame(sort_result[:1000], columns=["date", "id", "y"])
        df_save.to_csv(save_path)


if __name__ == '__main__':
    if args.run_type == "train":
        train()
    elif args.run_type == "test":
        test()
    else:
        raise ValueError(args.run_type)
