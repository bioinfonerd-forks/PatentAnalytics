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