from flask import Flask, request, render_template

from factory import Factory
from config import Config

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
    config = Config(DEBUG)

    keyword = None
    object = None
    predication = None

    if request.method == 'POST':
        # First search for subject
        try:
            abstract = request.form['abstract']
        except KeyError:
            error = 'No keyword entered or could be found'
            return render_template('query.html', error=error)

    return render_template('query.html')

if __name__ == "__main__":
    app.run()