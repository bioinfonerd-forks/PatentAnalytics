

class Classify(object):
    def __init__(self, config):
        self.config = config

    def train(self, feature_matrix, response_vector):
        """
        Train the model with the feature vector and response vector
        :return:
        """

        # Make sure the number of examples is greater than number of predictors
        if feature_matrix and response_vector:
            pass

