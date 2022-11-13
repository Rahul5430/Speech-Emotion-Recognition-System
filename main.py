# Print iterations progress
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import glob
# from glob import glob
import soundfile
from tqdm import tqdm
from IPython.display import Audio
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import librosa
import os
import speech_recognition as sr
import pandas as pd
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.io import wavfile as wav
import warnings
from python_speech_features import mfcc, logfbank


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# ### STEP 1 - IMPORT DEPENDENCIES LIBRARIES
get_ipython().magic('matplotlib inline')

warnings.simplefilter("ignore")

# ### STEP 2 - LOAD THE RAVDESS DATASET
os.listdir(path='./speech-emotion-recognition-ravdess-data')


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


dirName = './speech-emotion-recognition-ravdess-data'
listOfFiles = getListOfFiles(dirName)
len(listOfFiles)


def loadBasicEmotions(file, title):
    # LOAD FILE
    x, sr = librosa.load('./speech-emotion-recognition-ravdess-data/' + file)

    # DISPLAY WAVEPLOT
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(x, sr=sr)
    plt.title('Waveplot - ' + title)
    plt.savefig('Waveplot_' + title.replace(" ", "") + '.png')

    # PLAY AUDIO FILE
    sf.write('./speech-emotion-recognition-ravdess-data/' + file, x, sr)
    Audio(data=x, rate=sr)


emotionsData = {
    'Actor_01/03-01-01-01-01-01-01.wav': 'Male Neutral',
    'Actor_02/03-01-02-01-01-01-02.wav': 'Female Calm',
    'Actor_03/03-01-03-01-01-01-03.wav': 'Male Happy',
    'Actor_04/03-01-04-01-01-01-04.wav': 'Female Sad',
    'Actor_05/03-01-05-02-02-01-05.wav': 'Male Angry',
    'Actor_06/03-01-06-01-01-01-06.wav': 'Female Fearful',
    'Actor_07/03-01-07-01-01-01-07.wav': 'Male Disgust',
    'Actor_08/03-01-08-01-01-01-08.wav': 'Female Surprised',
}
for file, title in emotionsData.items():
    loadBasicEmotions(file, title)


# ### STEP 3 - USING SPEECH RECOGNITION API TO CONVERT AUDIO TO TEXT

# Use the Speech-Recognition API to get the Raw Text from Audio Files, Though Speech Recognition
# is less strong for large chunk of files , so used Error Handling , where when it is not be able to
# produce the text of a particular Audio File it prints the statement 'error'.Just for understanding Audio

r = sr.Recognizer()
for file in range(0, len(listOfFiles), 1):
    with sr.AudioFile(listOfFiles[file]) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
        except:
            print('error')

# ### STEP 4 -  PLOTTING TO UNDERSTAND RAW AUDIO FILES


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# Plotting the Basic Graphs for understanding of Audio Files :
for file in range(0, len(listOfFiles), 1):
    audio, sfreq = librosa.load(listOfFiles[file])
    time = np.arange(0, len(audio)) / sfreq

    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='Time(s)', ylabel='Sound Amplitude')

# PLOT THE SEPCTOGRAM
for file in range(0, len(listOfFiles), 1):
    sample_rate, samples = wavfile.read(listOfFiles[file])
    frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectogram)
    plt.imshow(spectogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# ### STEP 5 - VISUALIZATION OF AUDIO DATA
# Next Step is In-Depth Visualisation of Audio Files and its certain features to plot for.
# They are the Plotting Functions to be called later.


def plot_signals(signals):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(signals.keys())[i])
            axes[x, y].plot(list(signals.values())[i])
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fft(fft):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transform', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x, y].set_title(list(fft.keys())[i])
            axes[x, y].plot(freq, Y)
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(fbank.keys())[i])
            axes[x, y].imshow(list(fbank.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(
        nrows=2, ncols=5, sharex=False, sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Capstrum  Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x, y].set_title(list(mfccs.keys())[i])
            axes[x, y].imshow(list(mfccs.values())[i],
                              cmap='hot', interpolation='nearest')
            axes[x, y].get_xaxis().set_visible(False)
            axes[x, y].get_yaxis().set_visible(False)
            i += 1


def calc_fft(y, rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)


# Here The Data Set is loaded and plots are Visualised by Calling the Plotting Functions .
# printProgressBar(0, len(listOfFiles), prefix = 'Progress:', suffix = 'Complete')
for file in range(0, len(listOfFiles), 1):
    rate, data = wav.read(listOfFiles[file])
    fft_out = fft(data)
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(data, np.abs(fft_out))
    plt.show()
    # printProgressBar(file, len(listOfFiles), prefix='Progress:', suffix='Complete')

signals = {}
fft = {}
fbank = {}
mfccs = {}
# load data
for file in range(0, len(listOfFiles), 1):
    #     rate, data = wavfile.read(listOfFiles[file])
    signal, rate = librosa.load(listOfFiles[file], sr=44100)
    mask = envelope(signal, rate, 0.0005)
    signals[file] = signal
    fft[file] = calc_fft(signal, rate)

    bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
    fbank[file] = bank
    mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103).T
    mfccs[file] = mel

plot_signals(signals)
plt.show()

plot_fft(fft)
plt.show()

plot_fbank(fbank)
plt.show()

plot_mfccs(mfccs)
plt.show()

# ### STEP 6 - CLEANING & MASKING
# Now Cleaning Step is Performed where:
# DOWN SAMPLING OF AUDIO FILES IS DONE  AND PUT MASK OVER IT AND DIRECT INTO CLEAN FOLDER
# MASK IS TO REMOVE UNNECESSARY EMPTY VOIVES AROUND THE MAIN AUDIO VOICE


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10),  min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# The clean Audio Files are redirected to Clean Audio Folder Directory

for file in tqdm(glob.glob(r'./speech-emotion-recognition-ravdess-data/**/*.wav')):
    file_name = os.path.basename(file)
    signal, rate = librosa.load(file, sr=16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename=r'./clean_speech/'+str(file_name),
                  rate=rate, data=signal[mask])

# ### STEP 7 - FEATURE EXTRACTION
#
# Define a function extract_feature to extract the mfcc, chroma, and mel features from a sound file. This function takes 4 parameters- the file name and three Boolean parameters for the three features:
#
# **mfcc:** Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
#
# **chroma:** Pertains to the 12 different pitch classes
#
# **mel: Mel Spectrogram Frequency**
#
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

# ### STEP 8 - LABELS CLASSIFICATION
#
# Now, let’s define a dictionary to hold numbers and the emotions available in the RAVDESS dataset, and a list to hold those we want- calm, happy, fearful, disgust.


# Emotions in the RAVDESS dataset to be classified Audio Files based on .
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
# These are the emotions User wants to observe more :
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# ### STEP 9 - LOADING OF DATA & SPLITTING OF DATASET
#
# Now, let’s load the clean data with a function load_data() – this takes in the relative size of the test set as parameter. x and y are empty lists; we’ll use the glob() function from the glob module to get all the pathnames for the sound files in our dataset,


def load_data(test_size=0.33):
    x, y = [], []
    answer = 0
    for file in glob.glob(r'./clean_speech/*.wav'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            answer += 1
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append([emotion, file_name])
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Time to split the dataset into training and testing sets! Let’s keep the test set 25% of everything and use the load_data function for this.


# Split the dataset
x_train, x_test, y_trai, y_tes = load_data(test_size=0.25)
print(np.shape(x_train), np.shape(x_test), np.shape(y_trai), np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train), np.shape(y_test))
print(*test_filename, sep="\n")

# Get the shape of the training and testing datasets
# print((x_train.shape[0], x_test.shape[0]))
print((x_train[0], x_test[0]))
# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# ### STEP 10 - APPLY MLP CLASSIFIER
#
# Now, let’s Apply a MLPClassifier. This is a Multi-layer Perceptron Classifier; it optimizes the log-loss function using LBFGS or stochastic gradient descent. Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network for the purpose of classification. This is a feedforward ANN model.

# Apply Multi Layer Perceptron Classifier
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train, y_train)

# ### STEP 11 - SAVING THE MODEL

# SAVING THE MODEL
# Save the Modle to file in the current working directory
# For any new testing data other than the data in dataset

Pkl_Filename = "Speech_Emotions_Recognition_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)

# ### STEP 11 - LOAD THE SAVED MODEL

# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Speech_Emotions_Recognition_Model = pickle.load(file)

Speech_Emotions_Recognition_Model

# ### STEP 12 - PREDICT THE TEST DATA USING THE SAVED MODEL
#
# Let’s predict the values for the test set from saved model. This gives us y_pred (the predicted emotions for the features in the test set).

# predicting :
y_pred = Speech_Emotions_Recognition_Model.predict(x_test)

# ### STEP 13 - SUMMARIZATION OF PREDICTED DATA
#
# To calculate the accuracy of our model, we’ll call up the accuracy_score() function we imported from sklearn. Finally, we’ll round the accuracy to 2 decimal places and print it out.


results = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')
print(results)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Report")
print(classification_report(y_test, y_pred))

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

# ### STEP 14 - STORE THE PREDICTED FILE IN .CSV FILE

# Store the Prediction probabilities into CSV file
y_pred1 = pd.DataFrame(y_pred, columns=['predictions'])
y_pred1['file_names'] = test_filename
print(y_pred1)
y_pred1.to_csv('predictionfinal.csv')

# # END
