from pandas import DataFrame
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle


class Analyzer(object):
    def __init__(self, config):
        self.config = config
        self.corpus = None
        self.tfidf = None

    def load_data(self):
        """
        Load raw data from CSV
        :return:
        """
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        selected_data = df[((df.artunit.apply(str).str[:2] == "36") |
                            (df.artunit.apply(str).str[:2] == "24") |
                            (df.artunit.apply(str).str[:2] == "21"))]
        self.corpus = selected_data

    @staticmethod
    def tokenize(text):
        """

        :param text:
        :return:
        """
        tokens = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    @staticmethod
    def initialize_model(ngrams):
        """
        Create a new TFIDF model
        :param ngrams: Number of phrases to capture in model
        :return: The initialized TFIDF model
        """
        tfidf = TfidfVectorizer(
            ngram_range=(ngrams, ngrams),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            decode_error='ignore',
            tokenizer=Analyzer.tokenize
        )
        return tfidf

    def load_vector(self):
        """
        Load a saved model for additional training or testing
        :return:
        """
        self.tfidf = pickle.load(open('trained_vector.dill', 'rb'))

    def train(self, data, save=True):
        """
        Train an existing or new model with data
        :param corpus:
        :param save:
        :return:
        """
        # Do not initialize if the tfidf vector exists
        if self.tfidf:
            vec = self.tfidf
        else:
            vec = self.initialize_model()

        # Fit data to model
        tfs = vec.fit_transform(data)

        # Save vector if asked
        if save:
            pickle.dump(vec, open('trained_vector.dill', 'wb'))

        # Put the vector back into the object
        self.tfidf = vec

        # Return the feature matrix for these items
        return tfs

    def extract_features(self, corpus):
        """
         Get the feature matrix for a corpus with the saved vectorizer
        :param corpus:
        :return:
        """
        tfs = self.tfidf.transform(corpus)
        return tfs
