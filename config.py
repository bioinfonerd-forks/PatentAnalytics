import os


class Config(object):
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.model_name = '_feature_model.dill'
        self.matrix_name = '_feature_matrix.dill'
        self.art_units = (
            "36", "24", "21",
            # "16", "17", "26",
            # "28", "29", "37"
        )

    def get_model_path(self, feature_name):
        return os.path.join(self.data_dir, feature_name + self.model_name)

    def get_matrix_path(self, feature_name):
        return os.path.join(self.data_dir, feature_name + self.matrix_name)