from sklearn.externals import joblib
from sklearn.metrics import r2_score


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

    r2 = r2_score(validate_preds, validate_target)
    print(f"\n>> {model_name} R2: {r2}")


def all_predict(config_path):
    config = config_utils.read_config(config_path)
    print(">> data processing...")
    du = data_utils.DataUtils(config)
    train_input, train_target, validate_input, validate_target = du.get_train_and_validate_data()

    for model_name in config["models"]["all_model_name"]:
        try:
            model = joblib.load(config["models"]["model_path"] + f"/{model_name}.m")
            skm = model.sklearn_model(model_name)
            print(">> predict...\n")
            validate_preds = skm.predict(validate_input)

            print("## index: predict target")
            for index in range(len(validate_target)):
                if (index + 1) <= 10:
                    print(f">> {index}: {validate_preds[index]} {validate_target[index]}")
                else:
                    break

            r2 = r2_score(validate_preds, validate_target)
            print(f"\n>> {model_name} R2: {r2}")
        except Exception as e:
            print(e)


def test(config_path):
    config = config_utils.read_config(config_path)
    print(">> data processing...")
    du = data_utils.DataUtils(config)
    test_input = du.get_test_data()
    model_name = config["models"]["model_name"]
    model = joblib.load(config["models"]["model_path"] + f"/{model_name}.m")
    skm = model.sklearn_model(model_name)
    print(">> predict...\n")
    test_preds = skm.predict(test_input)
    print(test_preds)


if __name__ == '__main__':
    my_config_path = "./config.yaml"
    # train(my_config_path)
    # all_predict(my_config_path)
    test(my_config_path)
