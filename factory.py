from config import Config
from analyzer import Analyzer


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(config)

    def analyze_abstract_data(self, filename):
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data('abstract')
        self.analyzer.train_feature_model(1, 'abstract_model.dill')
        self.analyzer.save_features('abstract_feature_matrix.dill')

if __name__ == '__main__':
    config = Config()
    f = Factory(config)
    f.analyze_abstract_data('2016Patent_Data.csv')
