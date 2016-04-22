from pandas import DataFrame
import os
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle


class Analyzer(object):
    def __init__(self, config):
        self.config = config
        self.tfidf = None

    def load_data(self):
        df = DataFrame.from_csv(os.path.join(self.config.data_dir, '2016Patent_Data.csv'))
        selected_data = df[((df.artunit.apply(str).str[:2] == "36") | (df.artunit.apply(str).str[:2] == "24") | (df.artunit.apply(str).str[:2] == "21"))]
        return selected_data

    @staticmethod
    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stemmer = PorterStemmer()
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    # This is the vectorizer with are working with that currently uses unigrams and bigrams
    def initialize_vector(self):
        tfidf = TfidfVectorizer(
            ngram_range=(6, 6),
            stop_words='english',
            lowercase=True,
            strip_accents='ascii',
            decode_error='ignore',
            tokenizer=Analyzer.tokenize
        )
        return tfidf

    # Train a vectorizers and save resulting vectorizer in object
    def train(self, corpus, save=True):
        vec = self.initialize_vector()
        tfs = vec.fit_transform(corpus)
        print =('Vectorizer has been trained with', len(vec.vocabulary_.keys()), 'ngrams')
        if save:
            pickle.dump(vec, open('trained_vector.dill', 'wb'))
        self.tfidf = vec
        return tfs

    def load_vector(self):
        self.tfidf = pickle.load(open('trained_vector.dill', 'rb'))

    # Test a corpus with the saved vectorizer
    def test(self, corpus):
        tfs = self.tfidf.transform(corpus)
        return tfs
