from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import soundfile
import numpy as np
import librosa
import sys
import os
import warnings
warnings.simplefilter("ignore")

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

model = pickle.load(open('Speech_Emotions_Recognition_Model.pkl', 'rb'))


# Feature Extraction of Audio Files Function
# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(f, mfcc, chroma, mel):
    with soundfile.SoundFile(f) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


@app.route('/', methods=['GET'])
def hello():
    return jsonify('Hello world, use the /api route and read the README.md please')


@app.route('/api', methods=['GET', 'POST'])
def predict():
    # data = request.get_json(force=True)
    f = request.files.get('audioFile', None)
    # print(data, file=sys.stdout)
    print(f, file=sys.stdout)
    x = []
    x.append(extract_feature(f, mfcc=True, chroma=True, mel=True))
    prediction = model.predict(np.array(x))
    output = prediction[0]
    return jsonify(output)


port = int(os.environ.get('PORT', 5000))
debug = bool(os.environ.get('DEBUG', True))

if __name__ == '__main__':
    app.run(port=port, debug=debug)
