from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import dill as pickle
import os


class Classify(object):
    def __init__(self, config):
        self.config = config
        self.classifier = MultinomialNB(alpha=0.01)

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
        selected_feature_matrix = SelectKBest(chi2, k=50000).fit_transform(selected_feature_matrix, response_vector)
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

    def compare_classifiers(self, feature_matrix, response):
        """

        :return:
        """
        classifiers = [
            # KNeighborsClassifier(n_neighbors=2),
            MultinomialNB(alpha=0.01),
            SGDClassifier()
        ]
        for classifier in classifiers:
            self.classifier = classifier
            print(classifier)
            self.evaluate(feature_matrix, response)

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
        cross_val = KFold(len(response), n_folds=32, shuffle=True)
        for train_index, test_index in cross_val:
            X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
            y_train, y_test = response[train_index], response[test_index]

            self.train(X_train, y_train)
            score = self.classifier.score(X_test, y_test)
            print(score)

    def save_classifier(self):
        """

        :return:
        """
        # SAVE MODEL
        pickle.dump(self.classifier, open(os.path.join(self.config.data_dir, 'knn.dill'), 'wb'))

    def load_classifier(self):
        """

        :return:
        """
        self.classifier = pickle.load(open(os.path.join(self.config.data_dir, 'knn.dill'), 'rb'))