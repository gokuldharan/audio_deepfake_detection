import numpy as np
import librosa
import dataloader
import sklearn.mixture
from joblib import dump, load

SF = 16000
N_FFT = 512
N_FILES = 30

def fourierTransform(filenames):
    fourierTransforms = []

    for file in filenames:
        y, _ = librosa.load(file, sr=SF)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, win_length=N_FFT, hop_length = N_FFT//2, window='hamming', center=False)).T
        fourierTransforms.append(S)

    return fourierTransforms

def train(GenuineFiles, SpoofFiles):
    print("Starting Fourier Transform")
    X_trainGenuine = np.vstack(fourierTransform(GenuineFiles))
    X_trainSpoof = np.vstack(fourierTransform(SpoofFiles))

    print("Training Genuine GMM")
    genuineGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    genuineGMM.fit(X_trainGenuine)
    dump(genuineGMM, 'models/GMM/genuineGMM_n' + str(N_FILES) +'.joblib',compress=3)

    print("Training Spoof GMM")
    spoofGMM = sklearn.mixture.GaussianMixture(n_components=N_FFT, verbose=True, verbose_interval=1)
    spoofGMM.fit(X_trainSpoof)
    dump(spoofGMM, 'models/GMM/spoofGMM_n' + str(N_FILES) +'.joblib',compress=3)

def evaluate(XFiles):
    genuineGMM = load('models/GMM/genuineGMM_n' + str(N_FILES) +'.joblib')
    spoofGMM = load('models/GMM/spoofGMM_n' + str(N_FILES) +'.joblib')

    genuineScores = np.zeros(len(XFiles))
    spoofScores = np.zeros(len(XFiles))
    Y = np.zeros(len(XFiles))

    for i in range(len(XFiles)):
        #The first two arrays are just for debugging/analysis
        genuineScores[i] = genuineGMM.score(fourierTransform([XFiles[i]])[0])
        spoofScores[i] = spoofGMM.score(fourierTransform([XFiles[i]])[0])
        Y[i] = 1*(spoofScores[i] > genuineScores[i])

    return Y



def main():
    X_trainfilenames, Y_train, X_devfilenames, Y_dev, X_evalfilenames, Y_eval = dataloader.load_data()

    X_trainGenuineFiles = X_trainfilenames[0:N_FILES]
    Y_trainGenuine = Y_train[0:N_FILES]

    X_trainSpoofFiles = X_trainfilenames[-N_FILES:]
    Y_trainSpoof = Y_train[-N_FILES:]

    assert(Y_trainSpoof.all() and not Y_trainGenuine.any())

    train(X_trainGenuineFiles, X_trainSpoofFiles)
    Y_pred = evaluate(np.concatenate((X_trainGenuineFiles, X_trainSpoofFiles)))
    acc = np.mean(Y_pred == np.concatenate((Y_trainGenuine, Y_trainSpoof)))
    print("Training Set Accuracy: " + str(acc))

if __name__ == "__main__":
    main()