from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error

from utils import data_utils
from utils import config_utils
from models import sklearn_model


def train(config_path):
    config = config_utils.read_config(config_path)
    model_name = config["models"]["model_name"]

    print(">> data processing...")
    du = data_utils.DataUtils(config)
    train_input, train_target, validate_input, validate_target = du.get_train_and_validate_data()

    model = sklearn_model.Model()
    print(f">> run {model_name} model...")
    skm = model.sklearn_model(model_name)
    skm.fit(train_input, train_target)

    print(">> save model...")
    joblib.dump(model, config["models"]["model_path"] + f"/{model_name}.m")

    print(">> predict...\n")
    validate_preds = skm.predict(validate_input)

    print("## index: predict target")
    for index in range(len(validate_target)):
        if (index + 1) <= 10:
            print(f">> {index}: {validate_preds[index]} {validate_target[index]}")
        else:
            break

    mse = mean_squared_error(validate_preds, validate_target)
    print(f"\n>> {model_name} MSE: {mse}")


if __name__ == '__main__':
    my_config_path = "./config.yaml"
    train(my_config_path)
