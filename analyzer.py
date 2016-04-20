from pandas import DataFrame
import os
import re
from nltk.corpus import stopwords
# ...

class Analyzer(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        selected_data= df[((df.artunit.apply(str).str[:2]=="36") | (df.artunit.apply(str).str[:2]=="24") | (df.artunit.apply(str).str[:2]=="21"))]
        filtered_words = [word for word in selected_data.artunit.apply(str)[5] if word not in stopwords.words('english')]
        print(filtered_words)
        return df