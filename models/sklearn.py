from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm


def sklearn_model(model_name):
    model_dict = {
        "LinearRegression": linear_model.LinearRegression(),
        "HuberRegressor": linear_model.HuberRegressor(),
        "SGDRegressor": linear_model.SGDRegressor(),
        "GradientBoostingRegressor": ensemble.GradientBoostingRegressor(),
        "AdaBoostRegressor": ensemble.AdaBoostRegressor(),
        "RandomForestRegressor": ensemble.RandomForestRegressor(),
        "BaggingRegressor": ensemble.BaggingRegressor(),
        "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(),
        "SVR": svm.SVR(),
        "LinearSVR": svm.LinearSVR()
    }
    return model_dict[model_name]