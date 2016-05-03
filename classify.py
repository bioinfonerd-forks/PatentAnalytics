from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
import numpy as np
import dill as pickle
from results import Results


class Classify(object):
    def __init__(self, config, feature_matrix, response):
        self.config = config
        self.classifier = SGDClassifier(alpha=1.8*10**-6, l1_ratio=0.2)
        self.clf_name = None
        self.results = Results(config)
        self.feature_matrix = feature_matrix
        self.response = response
        self.classifiers = {
            'Bayes': [MultinomialNB(), {'alpha': np.arange(0.0001, 0.2, 0.0001)}],

            'SGD': [SGDClassifier(), {'alpha':  10**-6*np.arange(1, 15, 2),
                                      'l1_ratio': np.arange(0.05, 0.3, 0.05),
                                      'n_iter': [8], 'penalty': ['elasticnet']}],

            'Passive Aggressive': [PassiveAggressiveClassifier(), {'loss': ['hinge']}],

            'Perceptron': [Perceptron(), {'alpha': np.arange(0.00001, 0.001, 0.00001)}],

            'Tree': [DecisionTreeClassifier(), {'criterion': ['gini']}],
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

    def optimize_classifier(self, classifier, parameter_grid, parameter_of_interest):
        """

        :param feature_matrix:
        :param response:
        :param classifier:
        :param parameter_grid:
        :param parameter_of_interest:
        :return:
        """
        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        clf = GridSearchCV(classifier, parameter_grid, cv=cross_val, n_jobs=4)
        clf.fit(self.feature_matrix, self.response)
        print('Grid Search Completed', clf.best_estimator_, clf.best_score_)
        self.classifier = clf.best_estimator_
        self.results.plot_classifier_optimization(clf.grid_scores_, parameter_of_interest, parameter_of_interest)
        self.evaluate()

    def classifier_selection(self):
        """
        Select the classifier with the lowest error
        :return:
        """
        best_score = 0
        for clf_name in self.classifiers.keys():
            clf = self.classifiers[clf_name][0]
            score = self.evaluate_learning_curve(clf)
            print(clf_name, score)
            if score > best_score:
                self.clf_name = clf_name
                self.classifier = clf

    def evaluate_learning_curve(self, classifier):
        """
        Evaluate the classifier input with learning curve and cross validation
        :param feature_matrix:
        :param response:
        :return:
        """
        train_sizes = np.arange(100, int(0.9*len(self.response)), 5000)
        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        train_sizes, train_scores, valid_scores = learning_curve(classifier, self.feature_matrix, self.response,
                                                                 train_sizes=train_sizes, cv=cross_val,
                                                                 n_jobs=4)

        self.results.plot_learning_curve(train_sizes, train_scores, valid_scores, classifier)
        return np.mean(valid_scores[:-1])

    def evaluate(self):
        """

        :param feature_matrix:
        :param response:
        :return:
        """
        cross_val = KFold(len(self.response), n_folds=10, shuffle=True)
        scores = cross_validation.cross_val_score(self.classifier, self.feature_matrix, self.response, cv=cross_val)
        return np.mean(scores)

    def train(self, feature_matrix, response_vector):
        """
        Train the model with the feature vector and response vector
        :param feature_matrix: blh
        :param response_vector: blh
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        self.classifier.fit(feature_matrix, response_vector)

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