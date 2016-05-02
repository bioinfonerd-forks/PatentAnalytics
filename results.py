import matplotlib.pyplot as plt


class Results(object):
    def __init__(self, config):
        self.config = config

    @staticmethod
    def plot_clasifier_comparison(classifier_grid_scores):
        for classifier in classifier_grid_scores.keys():
            parms = [x.parameters['alpha'] for x in classifier_grid_scores[classifier]]
            scores = [x.mean_validation_score for x in classifier_grid_scores[classifier]]
            plt.plot(parms, scores, '.', label=classifier)

    @staticmethod
    def plot_learning_curve(train_sizes, train_scores, valid_scores):
        plt.plot(train_sizes, train_scores, label='Train Score')
        plt.plot(train_sizes, train_scores, label='CV Score')