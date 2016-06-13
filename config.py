import os
from datetime import date


class Config(object):
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.results_dir = os.path.join(self.base_dir, 'result')
        self.classifier_dir = os.path.join(self.base_dir, 'classifiers')
        self.model_name = '_model.dill'
        self.matrix_name = '_feature_matrix.dill'
        self.art_units = (
            "36", "24",
            # "21", "26"
            # "16", "17",
            # "28", "37"
        )
        self.today = date.today().isoformat()

    def get_model_path(self, feature_name):
        return os.path.join(self.data_dir, feature_name + "".join(self.art_units) + self.today + self.model_name)

    def get_matrix_path(self, feature_name):
        return os.path.join(self.data_dir, feature_name + "".join(self.art_units) + self.today + self.matrix_name)

    def get_classifier_path(self, clf_name, rand=True):
        return os.path.join(self.classifier_dir, clf_name + "".join(self.art_units) + self.today + '.dill')
