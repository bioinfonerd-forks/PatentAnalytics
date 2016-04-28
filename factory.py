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
        self.analyzer.extract_features(1, 'abstract')
        return self.analyzer.feature_matrix, self.analyzer.response

    def evalaute_performance(self, feature_matrix, response_vector):
        """

        :param feature_matrix:
        :param response_vector:
        :return:
        """
        c = Classify(self.config)
        c.evaluate(feature_matrix, response_vector)

    def train_classifier(self, feature_matrix, response_vector):
        """
        GET THE CLASSIFIER TRAINED
        :return:
        """
        c = Classify(self.config)
        # feature_matrix_reduced = c.reduce_dimensionality(feature_matrix.todense())
        c.train(feature_matrix, response_vector)

    def predict(self, abstract):
        """
        Predict group of a single abstract
        :param abstract:
        :return:
        """
        a = Analyzer(self.config)
        a.load_model('abstract')
        feature_vector = a.transform(abstract)

        c = Classify(self.config)
        c.load_classifier()

        group = c.predict(feature_vector)
        return group

if __name__ == '__main__':
    config_info = Config()
    f = Factory(config_info)
    feature_matrix, response_vector = f.analyze_abstract_data('2016Patent_Data.csv')
    f.evalaute_performance(feature_matrix, response_vector)