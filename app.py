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
    title = None
    abstract = None
    claims = None

    if request.method == 'POST':
        try:
            title = request.form['title']
        except KeyError:
            pass

        try:
            abstract = request.form['abstract']
        except KeyError:
            pass

        try:
            claims = request.form['claims']
        except KeyError:
            pass

    config = Config()
    f = Factory(config)

    if title:
        title_group = f.predict(title)

    if abstract:
        abstract_group = f.predict(abstract)

    if claims:
        claims_group = f.predict(claims)


    return render_template('query.html', abstract=abstract, group=group[0])

if __name__ == "__main__":
    app.run()