from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
import dill as pickle
import os


class Classify(object):
    def __init__(self, config):
        self.config = config
        self.classifier = None

    @staticmethod
    def reduce_dimensionality(feature_matrix):
        """

        :param feature_matrix: Dense nxp matrix
        :return: Reduced matrix with q*p features
        """
        pca = PCA(n_components=0.5*feature_matrix.shape[1])
        feature_matrix_reduced = pca.fit_transform(feature_matrix)
        return feature_matrix_reduced

    def train(self, feature_matrix, response_vector):
        """
        Train the model with the feature vector and response vector
        :param feature_matrix: blh
        :param response_vector: blh
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.classifier.fit(feature_matrix, response_vector)

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
        :param test_matrix:
        :param response:
        :return:
        """
        cross_val = KFold(len(response), n_folds=8)
        for train_index, test_index in cross_val:
            X_train, X_test = feature_matrix[train_index], feature_matrix[test_index]
            y_train, y_test = response[train_index], response[test_index]

            self.train(X_train, y_train)
            predictions = self.predict(X_test)

            false_count = 0
            true_count = 0

            for i, predicted_class in enumerate(predictions):
                if predicted_class == response[i]:
                    true_count += 1
                else:
                    false_count += 1

            print('True Classification percent:', true_count/len(predictions))
            print('False Classification percent:', false_count/len(predictions))

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