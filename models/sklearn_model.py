from sklearn import svm
from sklearn import ensemble
from sklearn import linear_model


class Model(object):
    def __init__(self):
        self.model_dict = {
            "SGDRegressor": linear_model.SGDRegressor(max_iter=1000),
            "HuberRegressor": linear_model.HuberRegressor(),
            "LinearRegression": linear_model.LinearRegression(),
            "LinearSVR": svm.LinearSVR(),
            "BaggingRegressor": ensemble.BaggingRegressor(),
            "AdaBoostRegressor": ensemble.AdaBoostRegressor(),
            "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(),
            "RandomForestRegressor": ensemble.RandomForestRegressor(),
            "GradientBoostingRegressor": ensemble.GradientBoostingRegressor()
        }

    def sklearn_model(self, model_name):
        return self.model_dict[model_name]
