import matplotlib.pyplot as plt


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_optimization(self, classifier_grid_scores):
        parms = [x.parameters['alpha'] for x in classifier_grid_scores]
        scores = [x.mean_validation_score for x in classifier_grid_scores]
        plt.plot(parms, scores, '.')

    def plot_classifier_comparison(self, learning_curves):
        pass

    def plot_learning_curve(self, train_sizes, train_scores, valid_scores):
        for classifier in train_scores.keys():
            plt.plot(train_sizes, train_scores[classifier], label=classifier + 'Train Score')
            plt.plot(train_sizes, valid_scores[classifier], label=classifier + 'CV Score')
        plt.show()
