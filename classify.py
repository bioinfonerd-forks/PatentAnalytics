from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


class Classify(object):
    def __init__(self, config):
        self.config = config

    @staticmethod
    def reduce_dimensionality(feature_matrix):
        """

        :param feature_matrix: Dense nxp matrix
        :return: Reduced matrix with q*p features
        """
        pca = PCA(n_components=0.5*feature_matrix.shape[1])
        feature_matrix_reduced = pca.fit_transform(feature_matrix)
        return feature_matrix_reduced

    @staticmethod
    def train(feature_matrix, response_vector):
        """
        Train the model with the feature vector and response vector
        :param feature_matrix: blh
        :param response_vector: blh
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(feature_matrix, response_vector)
        predictions = knn.predict(feature_matrix)

        false_count = 0
        true_count = 0

        for i, predicted_class in enumerate(predictions):
            if predicted_class == response_vector[i]:
                true_count += 1
            else:
                false_count += 1

        ('True Classification %i percent').format(true_count/len(predictions))
        ('False Classification %i percent').format(false_count/len(predictions))