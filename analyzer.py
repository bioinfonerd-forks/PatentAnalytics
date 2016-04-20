from pandas import DataFrame
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Analyzer(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        selected_data = df[((df.artunit.apply(str).str[:2] == "36") | (df.artunit.apply(str).str[:2] == "24") | (df.artunit.apply(str).str[:2] == "21"))]
        stop_words = set(stopwords.words('english'))
        for w in len(selected_data):
            selected_data['abstract'][w] = selected_data['abstract'][w].apply(str)
          #  something = word_tokenize(selected_data['abstract'][w])
           # selected_data['abstract'][w] = [x for x in something if not x in stop_words]
        print(selected_data)
        print(stop_words)
        return selected_data