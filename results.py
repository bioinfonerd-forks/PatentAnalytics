import matplotlib.pyplot as plt
import numpy as np


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_optimization(self, classifier_grid_scores, parameter):
        """
        Plot the results of the classifier optimization for a single classifier
        :param classifier_grid_scores:
        :param parameter:
        :return:
        """
        parms = [x.parameters[parameter] for x in classifier_grid_scores]
        scores = [x.mean_validation_score for x in classifier_grid_scores]
        plt.plot(parms, scores, '.')

    def plot_learning_curve(self, train_sizes, train_scores, valid_scores):
        """
        Plot the results from the learning curves of a single classifier
        :param train_sizes:
        :param train_scores:
        :param valid_scores:
        :return:
        """
        i = 0
        for classifier in train_scores.keys():
            plt.plot(train_sizes, np.mean(train_scores[classifier], axis=1), '.-', label=classifier + ' Train Score')
            plt.plot(train_sizes, np.mean(valid_scores[classifier], axis=1),  '.-', label=classifier + ' CV Score')
            i += 2
        plt.legend()
        plt.show()
