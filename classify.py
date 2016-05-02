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
import os
from results import Results
from sklearn.learning_curve import learning_curve


class Classify(object):
    def __init__(self, config):
        self.config = config
        self.classifier = None
        self.results = Results(config)
        self.classifiers = {
            'bayes': [MultinomialNB(), {'alpha': np.arange(0.001, 0.2, 0.001)}],
            'sgd': [SGDClassifier(), {'alpha': np.arange(0.00001, 0.0001, 0.00001),
                                      'l1_ratio': np.arange(0.5, 0.9, 0.1),
                                      'n_iter': [8], 'penalty': ['elasticnet']}],
            'passive_aggresive': [PassiveAggressiveClassifier(), {'loss': ['hinge']}],
            'perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}]
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
        selector = VarianceThreshold()
        selected_feature_matrix = selector.fit_transform(feature_matrix, response_vector)
        selected_feature_matrix = SelectKBest(chi2, k=int(0.9*feature_matrix.shape[1])).fit_transform(selected_feature_matrix, response_vector)
        return selected_feature_matrix

    def classifier_selection(self, feature_matrix, response):
        """

        :return:
        """

        cross_val = KFold(len(response), n_folds=10, shuffle=True)
        best_score = 0
        clf_results = dict()
        for classifier in self.classifiers.keys():
            clf = GridSearchCV(self.classifiers[classifier][0], self.classifiers[classifier][1], cv=cross_val)
            clf.fit(feature_matrix, response)
            if clf.best_score_ > best_score:
                self.classifier = clf.best_estimator_

            # Output results
            print(classifier, clf.best_params_, clf.best_score_)
            clf_results[classifier] = clf.best_params_

        # Grid results to results class
        self.results.plot_classifier_comparison(clf_results)

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

    def evaluate_classifiers(self, feature_matrix, response):
        """
        Evaluate the classifier
        :param feature_matrix:
        :param response:
        :return:
        """
        train_scores = dict()
        valid_scores = dict()
        train_sizes = np.arange(20000, 130000, 10000)
        cross_val = KFold(len(response), n_folds=10, shuffle=True)
        for classifier in self.classifiers:
            clf = self.classifiers[classifier]
            train_sizes, train_scores[classifier], valid_scores[classifier] = learning_curve(clf, feature_matrix, response,
                                                                                             train_sizes=train_sizes, cv=cross_val)

        self.results.plot_learning_curve(train_sizes, train_scores, valid_scores)

    def save_classifier(self, column_name):
        """
        :param column_name:
        :return:
        """
        # SAVE MODEL
        pickle.dump(self.classifier, open(os.path.join(self.config.data_dir, column_name + '_classifier.dill'), 'wb'))

    def load_classifier(self, column_name):
        """
        :param column_name:
        :return:
        """
        self.classifier = pickle.load(open(os.path.join(self.config.data_dir, column_name + '_classifier.dill'), 'rb'))