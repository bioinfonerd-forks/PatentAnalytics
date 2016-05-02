import matplotlib.pyplot as plt


class Results(object):
    def __init__(self, config):
        self.config = config

    def plot_classifier_comparison(self, classifier, classifier_grid_scores):
        parms = [x.parameters['alpha'] for x in classifier_grid_scores[classifier]]
        scores = [x.mean_validation_score for x in classifier_grid_scores[classifier]]
        plt.plot(parms, scores, '.', label=classifier)

    def plot_learning_curve(self, train_sizes, train_scores, valid_scores):
        plt.plot(train_sizes, train_scores, label='Train Score')
        plt.plot(train_sizes, valid_scores, label='CV Score')