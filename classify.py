from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
import numpy as np
import dill as pickle
from results import Results
from sklearn.learning_curve import learning_curve


class Classify(object):
    def __init__(self, config):
        self.config = config
        self.classifier = None
        self.clf_name = None
        self.results = Results(config)
        self.classifiers = {
            'Bayes': [MultinomialNB(), {'alpha': np.arange(0.0001, 0.2, 0.0001)}],
            'SGD': [SGDClassifier(), {'alpha': 10**-7 * np.arange(1, 10, 1),
                                      'l1_ratio': np.arange(0.1, 0.9, 0.1),
                                      'n_iter': [8], 'penalty': ['elasticnet']}],
            'Passive Aggressive': [PassiveAggressiveClassifier(), {'loss': ['hinge']}],
            'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}]
        }

    @staticmethod
    def reduce_dimensionality(feature_matrix):
        """

        :param feature_matrix: Dense nxp matrix
        :return: Reduced matrix with q*p features
        """
        pca = PCA(n_components=feature_matrix.shape[1])
        feature_matrix_reduced = pca.fit_transform(feature_matrix)
        return feature_matrix_reduced

    @staticmethod
    def feature_selection(feature_matrix, response_vector):
        """
        Remove zero variance features
        :param feature_matrix:
        :param response_vector:
        :return:
        """
        selected_feature_matrix = SelectKBest(chi2, k=int(0.8*feature_matrix.shape[0])).fit_transform(feature_matrix, response_vector)
        return selected_feature_matrix

    def optimize_classifier(self, feature_matrix, response, classifier, parameter_grid, parameter_of_interest):
        """

        :param feature_matrix:
        :param response:
        :param classifier:
        :param parameter_grid:
        :param parameter_of_interest:
        :return:
        """
        cross_val = KFold(len(response), n_folds=10, shuffle=True)
        clf = GridSearchCV(classifier, parameter_grid, cv=cross_val, n_jobs=2)
        clf.fit(feature_matrix, response)
        self.classifier = clf.best_estimator_
        self.evaluate_classifier(feature_matrix, response, self.classifier)
        self.results.plot_classifier_optimization(clf.grid_scores_, parameter_of_interest, self.clf_name)

    def classifier_selection(self, feature_matrix, response):
        """
        Select the classifier with the lowest error
        :return:
        """
        best_score = 0
        for clf_name in self.classifiers.keys():
            clf = self.classifiers[clf_name][0]
            score = self.evaluate_classifier(feature_matrix, response, clf)
            if score > best_score:
                self.clf_name = clf_name
                self.classifier = clf

    def evaluate_classifier(self, feature_matrix, response, classifier):
        """
        Evaluate the classifier input with learning curve and cross validation
        :param feature_matrix:
        :param response:
        :return:
        """
        train_sizes = np.arange(10000, len(response) - 30000, 10000)
        cross_val = KFold(len(response), n_folds=10, shuffle=True)
        train_sizes, train_scores, valid_scores = learning_curve(classifier, feature_matrix, response,
                                                                 train_sizes=train_sizes, cv=cross_val,
                                                                 n_jobs=2)

        self.results.plot_learning_curve(train_sizes, train_scores, valid_scores, classifier)
        return np.mean(valid_scores[:-1])

    def train(self, feature_matrix, response_vector):
        """
        Train the model with the feature vector and response vector
        :param feature_matrix: blh
        :param response_vector: blh
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        self.classifier.fit(feature_matrix, response_vector)

        # TODO Get classifier error

    def predict(self, test_matrix):
        """

        :param test_matrix:
        :return: Predictions
        """
        predictions = self.classifier.predict(test_matrix)
        return predictions

    def save_classifier(self):
        """
        :param column_name:
        :return:
        """
        # SAVE MODEL
        path = self.config.get_classifier_path(self.clf_name)
        pickle.dump(self.classifier, open(path, 'wb'))

    def load_classifier(self, column_name):
        """
        :param column_name:
        :return:
        """
        path = self.config.get_classifier_path(column_name)
        self.classifier = pickle.load(open(path, 'rb'))