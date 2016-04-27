from config import Config
from analyzer import Analyzer
from classify import Classify


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = None

    def analyze_abstract_data(self, filename):
        """
        Create the feature model and matrix for the abstract column
        :param filename:
        :return:
        """
        self.analyzer = Analyzer(self.config)
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data('abstract')
        self.analyzer.extract_features(1, 'abstract_model.dill')
        self.analyzer.save_features('abstract_feature_matrix.dill')
        return self.analyzer.feature_matrix, self.analyzer.response

    def train_classifier(self, feature_matrix, response_vector):
        """
        GET THE CLASSIFIER TRAINED
        :return:
        """
        c = Classify(self.config)
        c.train(feature_matrix, response_vector)


if __name__ == '__main__':
    config_info = Config()
    f = Factory(config_info)
    feature_matrix, response_vector = f.analyze_abstract_data('2016Patent_Data.csv')
    f.train_classifier(feature_matrix, response_vector)
