from flask import Flask, jsonify, redirect, url_for, request
from flask_cors import CORS
import main

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        query = request.form['query']
        # language = request.form['lang']
        tgt_sentences = request.form['tgt-sentences']
    return main.weblang(query, language, tgt_sentences)

if (__name__) == '__main__':
    app.run(debug = True)