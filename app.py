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
    #app.run(host='0.0.0.0', port=int(environ.get("PORT", 5000)))
    app.run(host='127.0.0.1')
