from sklearn.externals import joblib

from utils import data_utils
from utils import config_utils
from models.sklearn import linear_regression


def train(config_path):
    config = config_utils.read_config(config_path)

    du = data_utils.DataUtils(config)
    train_input, train_target, validate_input, validate_target = du.get_train_and_validate_data()

    model = linear_regression()
    model.fit(train_input, train_target)

    # save model
    joblib.dump(model, config["datas"]["model_path"] + "/sklearn-linear-regression.m")

    # validate
    validate_preds = model.predict(validate_input)

    for index in range(len(validate_target)):
        print(index, validate_preds[index], validate_target[index])
        if index > 20:
            break


if __name__ == '__main__':
    my_config_path = "./config.yaml"
    train(my_config_path)
