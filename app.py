from flask import Flask, request, render_template
#import sys
#from datetime import date
from factory import Factory
from config import Config


DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'
app = Flask(__name__)
app.config.from_object(__name__)
app.config['DEBUG'] = True
app.debug = True
app.config['SERVER_NAME'] = '0.0.0.0:5000'


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
        feature_vector = f.evaluate(title, abstract, claims)
        group = f.predict(feature_vector)
        return render_template('query.html', group=group)



if __name__ == '__main__':
    from os import environ
    port=environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    #app.run()
