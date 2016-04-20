from pandas import DataFrame
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Analyzer(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        w=1
        fulltext_without_stopwords = []
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        selected_data = df[((df.artunit.apply(str).str[:2] == "36") | (df.artunit.apply(str).str[:2] == "24") | (df.artunit.apply(str).str[:2] == "21"))]
        selected_data = selected_data.reset_index(drop=True)
        stop_words = stopwords.words('english')
        
        for w in range(len(selected_data)):
            fulltext_without_stopwords = ' '.join([x for x in word_tokenize(selected_data['abstract'][w]) if not x in stop_words])
            selected_data.set_value('abstract',w,fulltext_without_stopwords)
        print(selected_data)
        print(stop_words)
        return selected_data