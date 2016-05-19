from flask import Flask, request, render_template
import sys
from datetime import date
from factory import Factory
from config import Config
from classify import Classify
from results import Results
from analyzer import Analyzer
from scipy.sparse import hstack
from pandas import DataFrame
import os
from os import environ
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import dill as pickle
import numpy as np

DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'

app = Flask(__name__)
app.config.from_object(__name__)

@app.route('/')
def home():
    return render_template('query.html')


@app.route('/query', methods=['POST', 'GET'])
def submit_query():
    title = None
    abstract = None
    claims = None

    if request.method == 'POST':
        try:
            title = request.form['title']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            abstract = request.form['abstract']
        except KeyError:
            return render_template('query.html', error=KeyError)

        try:
            claims = request.form['claims']
        except KeyError:
            return render_template('query.html', error=KeyError)
            
    
    
    config = Config()
    f = Factory(config)
    return render_template('query.html', group=str(config.get_model_path('title'))
    
   
    #feature_vector = f.evaluate(title, abstract, claims)
    #group = f.predict(feature_vector)
    
    

if __name__ == '__main__':
    from os import environ
    #app.run(debug=False, host='0.0.0.0', port=environ.get("PORT", 5000))
    app.run()
