from pandas import DataFrame
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle


class Analyzer(object):
    def __init__(self, config):
        self.config = config
        self.data_frame = None
        self.data = None
        self.response = None
        self.feature_model = None
        self.feature_matrix = None

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
        self.data_frame = selected_data

    def extract_data(self, column_name):
        """

        :param column_name: Name list abstract or title or claims
        :return:
        """
        self.data = self.data_frame[column_name].tolist()
        self.response = self.data_frame['artunit'].tolist()

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
    def initialize_model(n_grams):
        """
        Create a new TFIDF model
        :param n_grams: Number of phrases to capture in model
        :return: The initialized TFIDF model
        """
        model = TfidfVectorizer(
            ngram_range=(n_grams, n_grams),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            decode_error='ignore',
            tokenizer=Analyzer.tokenize
        )
        return model

    def train_feature_model(self, n_grams, filename):
        """
        Train an existing or new model with data
        :param n_grams: Number of phrases to captures
        :return:
        """
        # TODO: Load specific feature model if one is specified

        # Use loaded data to train
        if not self.data:
            raise ReferenceError('Data not loaded into analyzer object')

        # Do not initialize if the tfidf vector exists
        if self.feature_model:
            vec = self.feature_model
        else:
            vec = self.initialize_model(n_grams)

        # Fit data to model
        tfs = vec.fit_transform(self.data)

        # Put the vector back into the object
        self.feature_model = vec

        # Return the feature matrix for these items
        self.feature_matrix = tfs

        # Save vector
        self.save_model(filename)

    def extract_features(self, corpus):
        """
        Get the feature matrix for a corpus with the saved vectorizer
        :param corpus:
        :return:
        """
        self.feature_matrix = self.feature_model.transform(corpus)

    def save_model(self, filename):
        """
        Save the feature model into a pickled object
        :return:
        """
        pickle.dump(self.feature_model, open(os.path.join(self.config.data_dir, filename), 'wb'))

    def load_model(self):
        """
        Load a saved model for additional training or testing
        :return:
        """
        self.feature_model = pickle.load(open(os.path.join(self.config.data_dir, 'trained_model.dill'), 'rb'))

    def save_features(self, filename):
        """
        Use pickle to save the feature matrix in a python object
        :return: Nothing
        """
        pickle.dump(self.feature_matrix, open(os.path.join(self.config.data_dir, filename), 'wb'))
