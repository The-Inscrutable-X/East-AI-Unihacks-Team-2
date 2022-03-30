from flask import Flask, jsonify, redirect, url_for, request
from flask_cors import CORS
from main import simple_weblang

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET', 'POST'])

def index():
    # return jsonify('{"asw": "ask"}')
    print(request.method)
    # if request.method == 'POST':
    # return jsonify(simple_weblang(request.form['query'], request.form['tgt-sentences']))
    return jsonify(simple_weblang('Guten tag'))

if (__name__) == '__main__':
    app.run(debug = True)