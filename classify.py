from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
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


class Classify(object):
    def __init__(self, config):
        self.config = config
        self.classifier = None
        self.results = Results(config)

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

    def classifier_selection(self, feature_matrix, response):
        """

        :return:
        """
        classifiers = {
            # 'knn':[KNeighborsClassifier(), {'n_neighbors': [1, 2, 3, 4, 5]}],
            'bayes': [MultinomialNB(), {'alpha': np.arange(0.001, 0.05, 0.001)}],
            'sgd': [SGDClassifier(), {'alpha': np.arange(0.001, 0.05, 0.001),
                                      'n_iter': [1, 2, 5, 8, 10]}]
        }

        cross_val = KFold(len(response), n_folds=10, shuffle=True)
        best_score = 0
        for classifier in classifiers.keys():
            clf = GridSearchCV(classifiers[classifier][0], classifiers[classifier][1], cv=cross_val)
            clf.fit(feature_matrix, response)
            print(classifier, clf.best_params_)
            if clf.best_score_ > best_score:
                self.classifier = clf.best_estimator_

    def predict(self, test_matrix):
        """

        :param test_matrix:
        :return: Predictions
        """
        predictions = self.classifier.predict(test_matrix)
        return predictions

    def evaluate(self, feature_matrix, response):
        """
        Evaluate the classifier
        :param feature_matrix:
        :param response:
        :return:
        """

        response = np.asarray(response)
        cross_val = KFold(len(response), n_folds=16, shuffle=True)
        score = []
        for train_index, test_index in cross_val:
            X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
            y_train, y_test = response[train_index], response[test_index]

            self.train(X_train, y_train)
            score = np.append(self.classifier.score(X_test, y_test), score)
        print('Classifier Mean Cross Val Score:', np.mean(score))

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