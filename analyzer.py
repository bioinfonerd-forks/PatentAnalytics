from pandas import DataFrame
import os


class Analyzer(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        return df