import matplotlib.pyplot as plt
import numpy as np


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_optimization(self, classifier_grid_scores):
        parms = [x.parameters['alpha'] for x in classifier_grid_scores]
        scores = [x.mean_validation_score for x in classifier_grid_scores]
        plt.plot(parms, scores, '.')

    def plot_learning_curve(self, train_sizes, train_scores, valid_scores):
        line_colors = plt.cm.Spectral(np.linspace(0,1,len(train_scores.keys())*2))
        i = 0
        for classifier in train_scores.keys():
            # plt.plot(train_sizes, np.mean(train_scores[classifier], axis=1), '.-', label=classifier + ' Train Score', c=line_colors[i])
            plt.plot(train_sizes, np.mean(valid_scores[classifier], axis=1),  '.-', label=classifier + ' CV Score', c=line_colors[i+1])
            i += 2
        plt.legend()
        plt.show()
