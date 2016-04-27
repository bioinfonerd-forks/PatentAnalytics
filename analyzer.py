from pandas import DataFrame
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle


class Analyzer(object):
    def __init__(self, config):
        self.config = config
        self.data = None
        self.feature_model = None
        self.features = None

    def load_patent_data(self, filename):
        """
        Load raw data from CSV
        :param filename: Name of csv file to extract patents
        :return:
        """
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, filename))
        selected_data = df[((df.artunit.apply(str).str[:2] == "36") |
                            (df.artunit.apply(str).str[:2] == "24") |
                            (df.artunit.apply(str).str[:2] == "21"))]
        self.data = selected_data

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
        model = TfidfVectorizer(
            ngram_range=(ngrams, ngrams),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            decode_error='ignore',
            tokenizer=Analyzer.tokenize
        )
        return model

    def load_vector(self):
        """
        Load a saved model for additional training or testing
        :return:
        """
        self.tfidf = pickle.load(open('trained_vector.dill', 'rb'))

    def train(self, ngrams, save=True):
        """
        Train an existing or new model with data
        :param ngrams: Number of phrases to captures
        :param save:
        :return:
        """
        # Use loaded data to train
        if not self.data:
            raise ReferenceError('Data not loaded into analyzer object')

        # Do not initialize if the tfidf vector exists
        if self.tfidf:
            vec = self.tfidf
        else:
            vec = self.initialize_model(ngrams)

        # Fit data to model
        tfs = vec.fit_transform(self.data)

        # Save vector if asked
        if save:
            pickle.dump(vec, open(os.path.join(self.config.data_dir, 'trained_vector.dill', 'wb')))

        # Put the vector back into the object
        self.feature_model = vec

        # Return the feature matrix for these items
        self.features = tfs

    def extract_features(self, corpus):
        """
         Get the feature matrix for a corpus with the saved vectorizer
        :param corpus:
        :return:
        """
        self.features = self.tfidf.transform(corpus)

    def save_features(self):
        """
        Use pickle to save the feature matrix in a python object
        :return: Nothing
        """
        pickle.dump(self.features, open(os.path.join(self.config.data_dir, 'features.py'), 'wb'))
