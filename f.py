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
    return jsonify(simple_weblang(request.args['query'], 'de', int(request.args['tgt-sentences'])))
    # return jsonify(simple_weblang())

if (__name__) == '__main__':
    app.run(debug = True)