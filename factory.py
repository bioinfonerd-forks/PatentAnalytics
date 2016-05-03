from config import Config
from analyzer import Analyzer
from classify import Classify
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(self.config)
        self.classify = Classify(self.config)

    def analyze_column_data(self, filename, column_name):
        """
        Create the feature model and matrix for the abstract column
        :param filename:
        :return:
        """
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data(column_name)
        n_grams = 3
        self.analyzer.extract_features(n_grams, column_name)
        return self.analyzer.feature_matrix, self.analyzer.response

    def compute_heuristics(self, filename, column_name):
        """
        Figure out what words make up the groups in the shit
        :param filename:
        :return:
        """
        self.analyze_column_data(filename, column_name)
        self.analyzer.heuristics(column_name)

    def evaluate_performance(self, feature_matrix, response_vector):
        """

        :param feature_matrix:
        :param response_vector:
        :return:
        """
        feature_matrix = self.classify.feature_selection(feature_matrix, response_vector)
        self.classify.classifier_selection(feature_matrix, response_vector)
        predicted_response = self.classify.predict(feature_matrix)
        confusion_matrix(response_vector, predicted_response)

    def optimize(self, feature_matrix, response_vector):
        """
        Optimize classifier
        :param feature_matrix:
        :param response_vector:
        :return:
        """
        feature_matrix = self.classify.feature_selection(feature_matrix, response_vector)
        print('Optimization Feature Selection')
        classifier = self.classify.classifiers['SGD'][0]
        parameters = self.classify.classifiers['SGD'][1]
        self.classify.optimize_classifier(feature_matrix, response_vector, classifier, parameters, 'l1_ratio')
        predicted_response = self.classify.predict(feature_matrix)
        confusion_matrix(response_vector, predicted_response)

    def full_train(self, feature_matrix, response_vector):
        """
        GET THE CLASSIFIER TRAINED
        :param feature_matrix:
        :param response_vector:
        :param column_name:
        :return:
        """
        feature_matrix = self.classify.feature_selection(feature_matrix, response_vector)
        self.classify.train(feature_matrix, response_vector)
        self.classify.save_classifier()

    def evaluate(self, title, abstract, claims):
        """
        Predict group of a single entry
        :param abstract:
        :return:
        """
        self.analyzer.load_model('title')
        title_vector = self.analyzer.transform(title)
        self.analyzer.load_model('abstract')
        abstract_vector = self.analyzer.transform(abstract)
        self.analyzer.load_model('claims')
        claims_vector = self.analyzer.transform(claims)

        feature_vector = hstack([title_vector, abstract_vector])
        feature_vector = hstack([feature_vector, claims_vector])
        return feature_vector

    def predict(self, feature_vector):
        """
        Predict class based on feature vector input
        :param feature_vector:
        :return:
        """
        self.classify.load_classifier('clf_name')
        group = self.classify.predict(feature_vector)
        return group

if __name__ == '__main__':
    config_info = Config()
    f = Factory(config_info)
    file = '2015_2016_Patent_Data_new.csv'

    # Get all the feature matrices
    title_matrix, response_vector = f.analyze_column_data(file, 'title')
    abstract_matrix, response_vector = f.analyze_column_data(file, 'abstract')
    claims_matrix, response_vector = f.analyze_column_data(file, 'claims')

    # Get them all together
    feature_matrix = hstack([title_matrix, abstract_matrix])
    feature_matrix = hstack([feature_matrix, claims_matrix])

    f.optimize(feature_matrix, response_vector)
    f.evaluate_performance(feature_matrix, response_vector)
    f.full_train(feature_matrix, response_vector)
    # f.compute_heuristics(file, column_name)
