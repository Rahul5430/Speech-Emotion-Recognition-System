from flask import Flask, request, jsonify
import pickle
import soundfile
import numpy as np
import librosa
import sys
import os
import warnings
warnings.simplefilter("ignore")

app = Flask(__name__)

model = pickle.load(open('Speech_Emotions_Recognition_Model.pkl', 'rb'))


# Feature Extraction of Audio Files Function
# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
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
    file = 'clean_speech/' + f.filename
    x = []
    x.append(extract_feature(file, mfcc=True, chroma=True, mel=True))
    prediction = model.predict(np.array(x))
    output = prediction[0]
    return jsonify(output)


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(port=port)
