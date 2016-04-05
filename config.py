import os


class Config(object):
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')