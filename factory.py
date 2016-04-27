from config import Config
from analyzer import Analyzer


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(config)


if __name__ == '__main__':
    config = Config()
    f = Factory(config)
    f.analyzer.load_data()
    f.analyzer.train()
