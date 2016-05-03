import matplotlib.pyplot as plt
import numpy as np


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_optimization(self, classifier_grid_scores, parameter, clf_name):
        """
        Plot the result of the classifier optimization for a single classifier
        :param classifier_grid_scores:
        :param parameter:
        :return:
        """
        fig = plt.figure()
        parms = [x.parameters[parameter] for x in classifier_grid_scores]
        scores = [x.mean_validation_score for x in classifier_grid_scores]
        plt.plot(parms, scores, '.')
        plt.title(clf_name)
        plt.show()

    def plot_learning_curves(self, train_sizes, valid_scores, classifiers):
        """
        Plot the result from the learning curves of a single classifier
        :param train_sizes:
        :param train_scores:
        :param valid_scores:
        :return:
        """
        for clf in classifiers:
            plt.plot(train_sizes[clf], np.mean(valid_scores[clf], axis=1),  '.-', label=clf)

        plt.legend(loc='best')
        plt.title('Classifier Learning Curve Comparison')
        plt.show()
