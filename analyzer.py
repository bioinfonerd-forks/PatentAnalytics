from pandas import DataFrame
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle
import numpy as np


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
                            (df.artunit.apply(str).str[:2] == "21") |
                            (df.artunit.apply(str).str[:2] == "16") |
                            (df.artunit.apply(str).str[:2] == "17") |
                            (df.artunit.apply(str).str[:2] == "26") |
                            (df.artunit.apply(str).str[:2] == "28") |
                            (df.artunit.apply(str).str[:2] == "37")
                            )]
        self.data_frame = selected_data

    def extract_data(self, column_name):
        """

        :param column_name: Name list abstract or title or claims
        :return:
        """
        self.data = self.data_frame[column_name].tolist()

        # Assign art unit to class
        self.data_frame['class'] = [0]*self.data_frame.shape[0]
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "36", 'class'] = 36
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "24", 'class'] = 24
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "21", 'class'] = 21
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "16", 'class'] = 16
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "17", 'class'] = 17
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "26", 'class'] = 26
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "28", 'class'] = 28
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "29", 'class'] = 29
        self.data_frame.loc[self.data_frame['artunit'].str[:2] == "37", 'class'] = 37

        self.response = self.data_frame['class'].tolist()

    @staticmethod
    def tokenize(text):
        """
        Tokenizer that uses porter stemmer to stemm all works
        :param text:
        :return:
        """
        tokens = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(item) for item in tokens]
        return stemmed

    @staticmethod
    def initialize_model(n_grams):
        """
        Create a new TFIDF model
        :param n_grams: Number of phrases to capture in model
        :return: The initialized TFIDF model
        """
        model = TfidfVectorizer(
            ngram_range=(1, n_grams),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            decode_error='replace',
            tokenizer=Analyzer.tokenize,
            norm='l2',
            min_df=5,
            max_features=100000
        )
        return model

    def train_feature_model(self, n_grams, feature_name):
        """
        Train an existing or new model with data
        :param n_grams: Number of phrases to captures
        :return:
        """

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

        # Print stats
        print('Feature matrix', self.feature_matrix.shape)

        # Save
        self.save_model(feature_name)
        self.save_features(feature_name)

    def extract_features(self, n_grams, feature_name):
        """
        Get the feature matrix for a corpus with the saved vectorizer
        :param corpus:
        :return:
        """
        if os.path.isfile(self.config.get_matrix_path(feature_name)):
            self.load_features(feature_name)
        elif os.path.isfile(self.config.get_model_path(feature_name)):
            self.load_model(feature_name)
            self.feature_matrix = self.feature_model.transform(self.data)
            self.save_features(feature_name)
        else:
            self.train_feature_model(n_grams, feature_name)

    def heuristics(self):
        """
        Determine the top words that govern selection to certain classes
        :return:
        """
        groups = np.unique(self.response)

        # Load feature model if not loaded
        if not self.feature_model:
            self.load_model('abstract')

        # Get model vocabulary
        vocabulary = np.asarray(self.feature_model.get_feature_names())

        for group in groups:
            group_features = self.feature_matrix[self.response == group, :]
            group_means = group_features.mean(axis=0)
            group_means[group_means > group_means.max()*0.9]

    def transform(self, data):
        """
        Transform single data entry for web app
        :param data:
        :return:
        """
        return self.feature_model.transform(data)

    def save_model(self, feature_name):
        """
        Save the feature model into a pickled object
        :param feature_name:
        :return:
        """
        pickle.dump(self.feature_model, open(self.config.get_model_path(feature_name), 'wb'))

    def load_model(self, feature_name):
        """
        Load a saved model for additional training or testing
        :param feature_name
        :return:
        """
        self.feature_model = pickle.load(open(self.config.get_model_path(feature_name), 'rb'))

    def save_features(self, feature_name):
        """
        Use pickle to save the feature matrix in a python object
        :param feature_name:
        :return: Nothing
        """
        pickle.dump(self.feature_matrix, open(self.config.get_matrix_path(feature_name), 'wb'))

    def load_features(self, feature_name):
        """

        :param feature_name:
        :return:
        """
        self.feature_matrix = pickle.load(open(self.config.get_matrix_path(feature_name), 'rb'))
