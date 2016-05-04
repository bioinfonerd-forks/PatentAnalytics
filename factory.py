from config import Config
from analyzer import Analyzer
from classify import Classify
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix


class Factory(object):
    def __init__(self, config):
        self.config = config
        self.analyzer = Analyzer(self.config)
        self.classify = None

    def analyze_column_data(self, filename, column_name):
        """
        Create the feature model and matrix for the abstract column
        :param filename:
        :return:
        """
        self.analyzer.load_patent_data(filename)
        self.analyzer.extract_data(column_name)
        n_grams = 3
        self.analyzer.extract_features(n_grams, column_name)
        return self.analyzer.feature_matrix, self.analyzer.response

    def compute_heuristics(self, filename, column_name):
        """
        Figure out what words make up the groups in the shit
        :param filename:
        :return:
        """
        self.analyze_column_data(filename, column_name)
        self.analyzer.heuristics(column_name)

    def evaluate_performance(self):
        """

        :param feature_matrix:
        :param response_vector:
        :return:
        """
        self.classify.feature_selection()
        self.classify.classifier_selection()


    def optimize(self, feature_matrix, response_vector):
        """
        Optimize classifier
        :param feature_matrix:
        :param response_vector:
        :return:
        """
        self.classify.feature_selection()
        self.classify.optimize_classifier('SGD')
        predicted_response = self.classify.predict(feature_matrix)
        print(confusion_matrix(response_vector, predicted_response))

    def full_train(self,):
        """
        GET THE CLASSIFIER TRAINED
        :param feature_matrix:
        :param response_vector:
        :param column_name:
        :return:
        """
        self.classify.train()
        self.classify.save_classifier()

    def evaluate(self, title, abstract, claims):
        """
        Predict group of a single entry
        :param abstract:
        :return:
        """
        self.analyzer.load_model('title')
        title_vector = self.analyzer.transform([title])
        self.analyzer.load_model('abstract')
        abstract_vector = self.analyzer.transform([abstract])
        self.analyzer.load_model('claims')
        claims_vector = self.analyzer.transform([claims])

        feature_vector = hstack([title_vector, abstract_vector])
        feature_vector = hstack([feature_vector, claims_vector])
        return feature_vector

    def predict(self, feature_vector):
        """
        Predict class based on feature vector input
        :param feature_vector:
        :return:
        """
        self.classify.load_classifier('clf_name')
        group = self.classify.predict(feature_vector)
        return group

    @staticmethod
    def get_all_column_data(file):
        """
        Combine all column data into a single feature matrix
        :param file:
        :return:
        """
        # Get all the feature matrices
        title_matrix, response_vector = f.analyze_column_data(file, 'title')
        abstract_matrix, response_vector = f.analyze_column_data(file, 'abstract')
        claims_matrix, response_vector = f.analyze_column_data(file, 'claims')

        # Get them all together
        feature_matrix = hstack([title_matrix, abstract_matrix])
        feature_matrix = hstack([feature_matrix, claims_matrix])
        return feature_matrix, response_vector

if __name__ == '__main__':
    # config_info = Config()
    # f = Factory(config_info)
    # file = '2015_2016_Patent_Data_new.csv'
    #
    # feature_matrix, response_vector = f.get_all_column_data(file)
    # f.classify = Classify(config_info, feature_matrix, response_vector)
    # f.evaluate_performance()
    # f.full_train()

    TITLE = 'SYSTEM AND METHOD FOR ESTIMATING THE POSITION AND ORIENTATION OF A MOBILE COMMUNICATIONS DEVICE IN A BEACON-BASED POSITIONING SYSTEM'
    ABSTRACT = 'An example of a lighting device including a light source, a modulator and a processor. The processor is configured to control the light source to emit light for general illumination and control the modulator to modulate the intensity of the emitted light to superimpose at least two sinusoids. Frequencies of the at least two sinusoids enable a mobile device to infer the physical location of the lighting device.'
    CLAIMS = '1. A lighting device, comprising: a light source; a modulator coupled to the light source; and a processor coupled to the modulator and configured to: control the light source to emit visible light for general illumination within a space; and control the modulator to: modulate the intensity of visible light emitted by the light source based on a signal comprising at least two superimposed sinusoids and in accordance with at least two frequencies of the at least two superimposed sinusoids such that the at least two superimposed sinusoids are simultaneously broadcast; vary the frequency of a first of the at least two superimposed sinusoids, between a number of varied frequencies and within a modulation range, during each of a plurality of cycles, each cycle corresponding to a timeframe; maintain each respective varied frequency of the first of the at least two superimposed sinusoids during a respective cycle for some period of time, each period of time being a fraction of the respective timeframe corresponding to the respective cycle such that the collection of time periods for the respective number of varied frequencies of the respective cycle equals the respective timeframe; and repeat the plurality of cycles some number of times.'

    config_info = Config()
    f = Factory(config_info)
    feature_vector = f.evaluate(TITLE, ABSTRACT, CLAIMS)
    f.predict(feature_vector)