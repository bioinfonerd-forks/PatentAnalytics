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
        f.classify.load_classifier('title')
        title_group = f.predict(title)

    if abstract:
        f.classify.load_classifier('abstract')
        abstract_group = f.predict(abstract)

    if claims:
        f.classify.load_classifier('claims')
        claims_group = f.predict(claims)

    return render_template('query.html', abstract=abstract, abstract_group=abstract_group[0],
                           title=title, title_group=title_group,
                           claims=claims, claims_group=claims_group)

if __name__ == "__main__":
    app.run()
