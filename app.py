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
    abstract = None

    if request.method == 'POST':
        # First search for subject
        try:
            abstract = request.form['abstract']
        except KeyError:
            error = 'No keyword entered or could be found'
            return render_template('query.html', error=error)

    if abstract:
        config = Config()
        f = Factory(config)
        group = f.predict(abstract)
        return render_template('query.html', abstract=abstract, group=group[0])
    return render_template('query.html')

if __name__ == "__main__":
    app.run()