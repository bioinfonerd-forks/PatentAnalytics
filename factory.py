from config import Config
from analyzer import Analyzer
from classify import Classify


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(self.config)
        self.classify = Classify(self.config)

    def analyze_abstract_data(self, filename):
        """
        Create the feature model and matrix for the abstract column
        :param filename:
        :return:
        """
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data('abstract')
        n_grams = 2
        self.analyzer.extract_features(n_grams, 'abstract')
        return self.analyzer.feature_matrix, self.analyzer.response

    def compute_heuristics(self, filename):
        """
        Figure out what words make up the groups in the shit
        :param filename:
        :return:
        """
        self.analyze_abstract_data(filename)
        self.analyzer.heuristics()

    def evaluate_performance(self, feature_matrix, response_vector):
        """

        :param feature_matrix:
        :param response_vector:
        :return:
        """
        self.classify.evaluate(feature_matrix, response_vector)

    def full_train(self, feature_matrix, response_vector):
        """
        GET THE CLASSIFIER TRAINED
        :return:
        """
        self.classify.train(feature_matrix, response_vector)
        self.classify.save_classifier()

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
    file = '2015_2016_Patent_Data.csv'
    f.compute_heuristics(file)
    # feature_matrix, response_vector = f.analyze_abstract_data(file)
    # f.evaluate_performance(feature_matrix, response_vector)