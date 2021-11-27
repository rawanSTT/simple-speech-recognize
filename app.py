import os

from flask import Flask, request, Response

import librosa
from keras.models import load_model
import numpy as np
import IPython.display as ipd

app = Flask(__name__)
model = load_model('static/best_model.hdf5')
classes = ["one", "zero", "two", "three"]


def cors_give_permission():
    resp = Flask.make_response(app, rv="\n")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-type'] = 'application/json'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
    resp.headers["Access-Control-Allow-Headers"] = "X-Requested-With, Authorization, Content-Type, Cache-Control"
    return resp


@app.route('/', methods=['OPTIONS'])
def give_permission():
    return cors_give_permission()


@app.route('/', methods=['POST'])
def analyze_view():
    response = analyze_voice(request.files['voice'])

    resp = Flask.make_response(app, rv=response)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-type'] = 'text/plain'

    return resp


def analyze_voice(voice_file):
    samples, sample_rate = librosa.load(voice_file, sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    ipd.Audio(samples, rate=8000)
    return predict(samples)


def predict(audio):
    prob = model.predict(audio.reshape(1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


if __name__ == '__main__':
    # app.run(debug=True, port=int(os.environ.get("PORT", 8000)))
    app.run()

